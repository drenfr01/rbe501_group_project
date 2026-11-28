"""`RobotBackend` for the physical SO-101, so the reach env and policies run unchanged on hardware."""

import logging
import math

import mujoco
import numpy as np

from rbe_501_so101.config import HardwareConfig, SimConfig
from rbe_501_so101.envs.protocols import ForwardKinematics
from rbe_501_so101.envs.types import JointLimits, RobotState
from rbe_501_so101.hardware.bus import JointBus
from rbe_501_so101.hardware.clock import Clock, SystemClock
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.safety import SafetyAbort, SafetyMonitor
from rbe_501_so101.hardware.types import MotorStatus

log = logging.getLogger(__name__)


class LeRobotSo101Backend:
    """Drives a `JointBus` at the sim control rate, in sim radians.

    Commands integrate on `q_cmd` and are clipped to the tighter of the sim and
    calibrated real limits, minus `limit_margin_rad`. Every command passes the
    `SafetyMonitor` before it is sent; on any abort the last sent command is
    held and the arm ramps to rest before torque is cut. Velocity is the
    difference of encoder positions over the control period, and `tcp_pos` is
    forward kinematics of the measured angles (encoder-based, not external).
    """

    def __init__(
        self,
        bus: JointBus,
        joint_map: JointMap,
        safety: SafetyMonitor,
        kinematics: ForwardKinematics,
        model: mujoco.MjModel,
        sim_cfg: SimConfig = SimConfig(),
        hw_cfg: HardwareConfig = HardwareConfig(),
        clock: Clock | None = None,
        home_deg: tuple[float, ...] | None = None,
    ) -> None:
        self._bus = bus
        self._map = joint_map
        self._safety = safety
        self._fk = kinematics
        self._hw = hw_cfg
        self._clock = clock or SystemClock()
        self._dt = sim_cfg.control_dt
        self._n_arm = len(sim_cfg.arm_joints)
        self._home = np.deg2rad(home_deg or sim_cfg.initial_deg)
        self._rest = np.deg2rad(hw_cfg.rest_deg)
        self._limits = self._clipped_limits(model, sim_cfg)
        self._q_cmd = self._home.copy()
        self._last_sent = self._home.copy()
        self._prev_q: np.ndarray | None = None
        self._deadline = 0.0
        self._steps = 0
        self._torque_on = False
        self.overrun_count = 0
        self.status_log: list[tuple[int, tuple[MotorStatus, ...]]] = []

    def reset(self, rng: np.random.Generator) -> RobotState:
        """Enable torque in place (first time), ramp to home, settle, and report the state."""
        if not self._torque_on:
            self._hold_present()
        self._ramp_guarded(self._home, self._hw.home_ramp_s)
        self._clock.sleep(self._hw.settle_s)
        q = self._read()
        self._prev_q = q
        self._deadline = self._clock.monotonic()
        return RobotState(qpos=q, qvel=np.zeros_like(q), q_cmd=self._q_cmd.copy(), tcp_pos=self._fk.forward(q))

    def apply_delta(self, delta_arm_rad: np.ndarray) -> None:
        """Integrate a relative arm command and clip it to the safe limits."""
        n = self._n_arm
        self._q_cmd[:n] = np.clip(self._q_cmd[:n] + delta_arm_rad, self._limits.lower, self._limits.upper)

    def step(self) -> RobotState:
        """Check, send, wait out the period, then read; abort safely on any violation."""
        try:
            return self._step_unguarded()
        except SafetyAbort as exc:
            log.error("safety abort: %s", exc)
            self._shutdown()
            raise

    def joint_limits(self) -> JointLimits:
        """Safe arm limits (tighter of sim and real, minus the margin)."""
        return self._limits

    def close(self) -> None:
        """Ramp to rest, cut torque, and disconnect."""
        if self._torque_on:
            self._shutdown()
        self._bus.close()

    def _step_unguarded(self) -> RobotState:
        """One control period; raises `SafetyAbort` without cleaning up."""
        self._send(self._q_cmd)
        self._wait_for_deadline()
        q = self._read()
        qvel = (q - self._prev_q) / self._dt if self._prev_q is not None else np.zeros_like(q)
        self._prev_q = q
        self._steps += 1
        if self._steps % self._safety.status_check_every_steps == 0:
            self._check_status()
        return RobotState(qpos=q, qvel=qvel, q_cmd=self._q_cmd.copy(), tcp_pos=self._fk.forward(q))

    def _check_status(self) -> None:
        """Read, log, and safety-check motor status."""
        status = self._bus.read_status()
        self.status_log.append((self._steps, status))
        self._safety.check_status(status)

    def _wait_for_deadline(self) -> None:
        """Sleep to a drift-free monotonic deadline; count and report overruns."""
        self._deadline += self._dt
        remaining = self._deadline - self._clock.monotonic()
        overran = remaining < 0
        if overran:
            self.overrun_count += 1
            log.warning("control loop overran by %.1f ms", -1000 * remaining)
            self._deadline = self._clock.monotonic()
        else:
            self._clock.sleep(remaining)
        self._safety.record_timing(overran)

    def _hold_present(self) -> None:
        """Make the present pose the goal before enabling torque, so nothing jumps."""
        present = self._bus.read_positions()
        self._bus.send_positions(present)
        self._bus.set_torque(True)
        self._torque_on = True
        self._q_cmd = self._map.to_sim(present)
        self._last_sent = self._q_cmd.copy()

    def _ramp_guarded(self, goal: np.ndarray, duration_s: float) -> None:
        """Ramp with cleanup on abort (used for the home ramp)."""
        try:
            self._ramp(goal, duration_s)
        except SafetyAbort as exc:
            log.error("safety abort during ramp: %s", exc)
            self._shutdown()
            raise

    def _ramp(self, goal: np.ndarray, duration_s: float) -> None:
        """Linearly interpolate the full command to `goal` over `duration_s`."""
        start = self._q_cmd.copy()
        n = max(1, math.ceil(duration_s / self._dt))
        for k in range(1, n + 1):
            self._q_cmd = start + (goal - start) * k / n
            self._send(self._q_cmd)
            self._clock.sleep(self._dt)

    def _shutdown(self) -> None:
        """Ramp from the last sent command to rest; cut torque only if the ramp completed."""
        self._q_cmd = self._last_sent.copy()
        try:
            self._ramp(self._rest, self._hw.rest_ramp_s)
        except SafetyAbort as exc:
            log.error("rest ramp stopped (%s); torque left ON, support the arm before cutting power", exc)
            return
        self._bus.set_torque(False)
        self._torque_on = False

    def _send(self, q: np.ndarray) -> None:
        """Safety-check a full command and send it in LeRobot units."""
        self._safety.check_command(q)
        self._bus.send_positions(self._map.to_lerobot(q))
        self._last_sent = q.copy()

    def _read(self) -> np.ndarray:
        """Measured joint angles in sim radians."""
        return self._map.to_sim(self._bus.read_positions())

    def _clipped_limits(self, model: mujoco.MjModel, sim_cfg: SimConfig) -> JointLimits:
        """Tighter of the sim joint ranges and the calibrated real ranges, shrunk by the margin."""
        joint_ids = [model.joint(j).id for j in sim_cfg.arm_joints]
        sim_low, sim_high = model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1]
        real = self._map.arm_limits_to_sim(self._bus.limits_deg())
        margin = self._hw.limit_margin_rad
        return JointLimits(
            lower=np.maximum(sim_low, real.lower) + margin, upper=np.minimum(sim_high, real.upper) - margin
        )

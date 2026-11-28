"""Real-servo effects in sim: stiction as play on the servo goal, and LeRobot's relative goal clip.

The real STS3215 behaves like a stiff P controller with static friction: the
joint only moves once the position error pushes harder than friction, so it
stops one friction band short of the goal on the side it came from. In
quasi-static terms that is play (backlash) on the goal the actuator acts on,
`goal = clip(goal, command - band, command + band)`, which is what this
wrapper feeds MuJoCo's position actuator. MuJoCo's own `frictionloss` cannot
do this; it creeps to the same equilibrium from both sides (D-03).
"""

from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np

from rbe_501_so101.config import ServoEffectsConfig
from rbe_501_so101.envs.backend_wrapper import BackendWrapper
from rbe_501_so101.envs.calibration import ActuatorCalibration
from rbe_501_so101.envs.protocols import RobotBackend
from rbe_501_so101.envs.types import JointLimits, RobotState


class TorqueModel(Protocol):
    """Gravity holding torque per arm joint at a full joint vector."""

    def torque(self, q: np.ndarray) -> np.ndarray:
        """Arm-joint torque (N*m)."""
        ...


@dataclass(frozen=True)
class StictionParams:
    """Per-arm-joint friction half-band `half_band_rad + load_slope_rad_per_nm * |gravity torque|`."""

    half_band_rad: np.ndarray
    load_slope_rad_per_nm: np.ndarray

    @classmethod
    def from_calibration(cls, calibration: ActuatorCalibration, joints: tuple[str, ...]) -> Self:
        """Read the band from a calibration; joints without one have no stiction."""
        entries = {e.joint: e for e in calibration.joints}
        band = [getattr(entries.get(j), "stiction_half_band_rad", None) or 0.0 for j in joints]
        slope = [getattr(entries.get(j), "stiction_load_slope_rad_per_nm", None) or 0.0 for j in joints]
        return cls(np.array(band, dtype=float), np.array(slope, dtype=float))

    @property
    def has_load_slope(self) -> bool:
        """True if any band depends on the gravity load."""
        return bool(np.any(self.load_slope_rad_per_nm != 0.0))


def play(goal: np.ndarray, command: np.ndarray, half_band: np.ndarray) -> np.ndarray:
    """Backlash operator: the goal moves only as far as needed to stay within the band of the command."""
    return np.clip(goal, command - half_band, command + half_band)


def hysteresis(
    goal: np.ndarray, prev_command: np.ndarray, command: np.ndarray, half_band: np.ndarray, ratio: float
) -> np.ndarray:
    """Play with a smooth (Dahl-type) corner: the new goal after the command moves `prev_command -> command`.

    With lag `z = command - goal` and `u = 1 - sign(move) * z / band` (0 when
    trailing by the full band), the lag grows one-for-one with the move while
    `u > ratio`, then `u` decays as `exp(-move / (ratio * band))`. Ratio 0 is
    `play`; larger ratios make the goal start slipping earlier after a reversal.
    """
    if ratio <= 0.0:
        return play(goal, command, half_band)
    move = command - prev_command
    sign = np.where(move < 0.0, -1.0, 1.0)
    band = np.maximum(half_band, np.finfo(float).tiny)
    u0 = np.clip(1.0 - sign * (prev_command - goal) / band, 0.0, 2.0)
    dist, linear = np.abs(move), np.maximum(u0 - ratio, 0.0) * band
    u = np.where(dist <= linear, u0 - dist / band, np.minimum(u0, ratio) * np.exp(-(dist - linear) / (ratio * band)))
    return np.where(half_band > 0.0, command - sign * band * (1.0 - u), command)


def _tightened(limits: JointLimits, window: tuple[tuple[float, float], ...] | None) -> JointLimits:
    """Intersection of joint limits with a per-joint (low, high) window."""
    if window is None:
        return limits
    low, high = np.array(window, dtype=float).T
    return JointLimits(lower=np.maximum(limits.lower, low), upper=np.minimum(limits.upper, high))


class ServoEffectsBackend(BackendWrapper):
    """Feeds the inner backend the goal a real servo would act on.

    The controller's command is integrated here and reported as `q_cmd`, as on
    hardware. Each step the command is clipped around the last measured
    position (LeRobot), then passed through the friction hysteresis before it
    reaches the inner actuator. `command_window` (sim radians per arm joint)
    tightens the command limits to the real arm's, as the real backend does.
    """

    def __init__(
        self,
        inner: RobotBackend,
        params: StictionParams,
        cfg: ServoEffectsConfig,
        gravity: TorqueModel | None = None,
        command_window: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        super().__init__(inner)
        if cfg.stiction and params.has_load_slope and gravity is None:
            raise ValueError("a load-dependent stiction band needs a gravity torque model")
        self._params = params
        self._cfg = cfg
        self._gravity = gravity
        self._limits = _tightened(inner.joint_limits(), command_window)
        self._scale = 1.0
        self._command = np.zeros(0)
        self._goal = np.zeros(0)
        self._target = np.zeros(0)
        self._inner_cmd = np.zeros(0)
        self._q = np.zeros(0)

    @property
    def command(self) -> np.ndarray:
        """The controller's command (arm joints, then gripper), as a copy."""
        return np.append(self._command, self._inner_cmd[len(self._command):])

    def reset(self, rng: np.random.Generator) -> RobotState:
        """Reset the inner arm, start the goal at its command, and draw this episode's band scale."""
        state = self.inner.reset(rng)
        n = len(self.inner.joint_limits().lower)
        self._scale = float(rng.uniform(*self._cfg.stiction_scale_range))
        self._command = state.q_cmd[:n].copy()
        self._goal = self._command.copy()
        self._target = self._command.copy()
        return self._observe(state)

    def joint_limits(self) -> JointLimits:
        """Inner limits, tightened to the command window if one was given (as the real backend's are)."""
        return self._limits

    def apply_delta(self, delta_arm_rad: np.ndarray) -> None:
        """Integrate the controller's command within the joint limits."""
        limits = self._limits
        self._command = np.clip(self._command + np.asarray(delta_arm_rad, dtype=float), limits.lower, limits.upper)

    def step(self) -> RobotState:
        """Send the servo goal for this period to the inner backend and advance it."""
        self._goal = self._servo_goal()
        self.inner.apply_delta(self._goal - self._inner_cmd[: len(self._goal)])
        return self._observe(self.inner.step())

    def _servo_goal(self) -> np.ndarray:
        """Relative clip around the measured position, then the friction hysteresis."""
        n = len(self._command)
        target = self._command
        if self._cfg.relative_clip_rad is not None:
            clip = self._cfg.relative_clip_rad
            target = np.clip(target, self._q[:n] - clip, self._q[:n] + clip)
        if not self._cfg.stiction:
            return target.copy()
        goal = hysteresis(self._goal, self._target, target, self._half_band(), self._cfg.stiction_transition_ratio)
        self._target = target.copy()
        return goal

    def _half_band(self) -> np.ndarray:
        """This episode's band at the current gravity load."""
        band = self._params.half_band_rad
        if self._params.has_load_slope:
            band = band + self._params.load_slope_rad_per_nm * np.abs(self._gravity.torque(self._q))
        return self._scale * band

    def _observe(self, state: RobotState) -> RobotState:
        """Remember what the inner arm did and report the controller's command."""
        self._inner_cmd = state.q_cmd.copy()
        self._q = state.qpos.copy()
        return RobotState(qpos=state.qpos, qvel=state.qvel, q_cmd=self.command, tcp_pos=state.tcp_pos)

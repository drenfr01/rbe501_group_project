"""A `JointBus` (and `Clock`) backed by the MuJoCo arm, so hardware procedures run in sim."""

import numpy as np

from rbe_501_so101.config import SimConfig
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend
from rbe_501_so101.hardware.bus import clip_relative
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.types import MotorSettings, MotorStatus

SIM_RNG_SEED = 0
NOT_MEASURED = float("nan")


class SimJointBus:
    """Speaks LeRobot units to a simulated arm; sleeping advances the physics.

    Readings are rounded to `quantization_rad` like an encoder. Only the arm
    joints are commanded; the gripper stays at the backend's home angle. The
    sim has no thermal or electrical model, so status fields are NaN. With
    `max_relative_deg`, arm goals are clipped around the present reading as
    LeRobot does (the uncommanded gripper is not, since it never moves).
    """

    def __init__(
        self,
        backend: MujocoSo101Backend,
        joint_map: JointMap,
        sim_cfg: SimConfig = SimConfig(),
        quantization_rad: float = 0.0,
        max_relative_deg: float | None = None,
    ) -> None:
        self._backend = backend
        self._map = joint_map
        self._max_relative_deg = max_relative_deg
        self._dt = sim_cfg.control_dt
        self._quant = quantization_rad
        self._n_arm = len(sim_cfg.arm_joints)
        self._state = backend.reset(np.random.default_rng(SIM_RNG_SEED))
        self._time = 0.0
        self._pending_s = 0.0

    def monotonic(self) -> float:
        """Simulated time in seconds."""
        return self._time

    def sleep(self, seconds: float) -> None:
        """Advance whole control periods covering `seconds` (remainders carry over)."""
        self._pending_s += max(seconds, 0.0)
        while self._pending_s >= self._dt - 1e-12:
            self._state = self._backend.step()
            self._pending_s -= self._dt
            self._time += self._dt

    def read_positions(self) -> dict[str, float]:
        """Encoder-rounded joint positions in LeRobot units."""
        return self._map.to_lerobot(self._round(self._state.qpos))

    def read_status(self) -> tuple[MotorStatus, ...]:
        """Placeholder status: the sim does not model temperature, voltage, or load."""
        return tuple(MotorStatus(j, NOT_MEASURED, NOT_MEASURED, NOT_MEASURED) for j in self._map.joints)

    def read_settings(self) -> tuple[MotorSettings, ...]:
        """The sim has no servo registers."""
        return ()

    def send_positions(self, goal: dict[str, float]) -> dict[str, float]:
        """Move the arm command to the (clipped) goal; missing joints keep their command."""
        gripper = pos_key(self._map.joints[-1])
        arm_goal = {k: v for k, v in goal.items() if k != gripper}
        sent = {**goal, **clip_relative(arm_goal, self.read_positions(), self._max_relative_deg)}
        current = self._map.to_lerobot(self._backend.command)
        q = self._map.to_sim({**current, **sent})
        self._backend.apply_delta(q[: self._n_arm] - self._backend.command[: self._n_arm])
        return sent

    def set_torque(self, enabled: bool) -> None:
        """The simulated servos are always powered."""

    def limits_deg(self) -> dict[str, tuple[float, float]]:
        """Model joint limits converted to LeRobot units."""
        limits = self._backend.joint_limits()
        n = self._n_arm
        low = self._map.to_lerobot(np.append(limits.lower, self._backend.command[n]))
        high = self._map.to_lerobot(np.append(limits.upper, self._backend.command[n]))
        return {k.removesuffix(".pos"): (min(low[k], high[k]), max(low[k], high[k])) for k in low}

    def close(self) -> None:
        """Release the backend."""
        self._backend.close()

    def _round(self, q: np.ndarray) -> np.ndarray:
        """Snap to the encoder grid when quantization is enabled."""
        return np.round(q / self._quant) * self._quant if self._quant > 0 else q.copy()

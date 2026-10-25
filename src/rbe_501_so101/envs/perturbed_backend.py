"""Wrap any RobotBackend with encoder rounding and whole-step command delay."""

from collections import deque
from typing import Any

import numpy as np

from rbe_501_so101.config import PerturbationConfig
from rbe_501_so101.envs.protocols import ForwardKinematics, Renderable, RobotBackend, TargetMarker
from rbe_501_so101.envs.types import JointLimits, RobotState


class PerturbedBackend:
    """Makes a simulated arm look like the real one to the policy.

    Delay: deltas queue for `delay_steps` control periods before reaching the
    inner backend. The reported `q_cmd` is the command the controller has
    issued (including in-flight deltas), as it would be on hardware.
    Rounding: `qpos` snaps to `quantization_rad`, `qvel` is the difference of
    rounded positions over the control period, and `tcp_pos` is recomputed by
    forward kinematics from the rounded `qpos`, as the real backend does.
    Sub-step delays cannot be represented.
    """

    def __init__(
        self, inner: RobotBackend, cfg: PerturbationConfig, control_dt: float, kinematics: ForwardKinematics
    ) -> None:
        self.inner = inner
        self._cfg = cfg
        self._dt = control_dt
        self._fk = kinematics
        self.delay_steps = cfg.delay_steps
        self._queue: deque[np.ndarray] = deque()
        self._pending: np.ndarray | None = None
        self._prev_q: np.ndarray | None = None

    @property
    def model(self) -> Any:
        """Inner MuJoCo model, if any (lets the env share it with samplers and IK)."""
        return getattr(self.inner, "model", None)

    @property
    def ids(self) -> Any:
        """Inner model indices, if any."""
        return getattr(self.inner, "ids", None)

    def reset(self, rng: np.random.Generator) -> RobotState:
        """Reset the inner backend, draw this episode's delay, and clear the queue."""
        state = self.inner.reset(rng)
        if self._cfg.randomize_delay:
            self.delay_steps = int(rng.integers(0, self._cfg.delay_steps + 1))
        n_arm = len(self.inner.joint_limits().lower)
        self._queue = deque(np.zeros(n_arm) for _ in range(self.delay_steps))
        self._pending = np.zeros(n_arm)
        self._prev_q = None
        return self._perturb(state)

    def apply_delta(self, delta_arm_rad: np.ndarray) -> None:
        """Accumulate this step's command change."""
        self._pending = self._pending + np.asarray(delta_arm_rad, dtype=float)

    def step(self) -> RobotState:
        """Release the oldest queued delta to the inner backend and advance one period."""
        self._queue.append(self._pending)
        self._pending = np.zeros_like(self._pending)
        self.inner.apply_delta(self._queue.popleft())
        return self._perturb(self.inner.step())

    def joint_limits(self) -> JointLimits:
        """Inner limits."""
        return self.inner.joint_limits()

    def close(self) -> None:
        """Close the inner backend."""
        self.inner.close()

    def set_target_marker(self, xyz: np.ndarray) -> None:
        """Forward to the inner backend if it has a marker."""
        if isinstance(self.inner, TargetMarker):
            self.inner.set_target_marker(xyz)

    def render(self) -> np.ndarray | None:
        """Forward to the inner backend if it renders."""
        return self.inner.render() if isinstance(self.inner, Renderable) else None

    def is_open(self) -> bool:
        """Inner viewer state (True when the inner backend cannot render)."""
        return self.inner.is_open() if isinstance(self.inner, Renderable) else True

    def _perturb(self, state: RobotState) -> RobotState:
        """Apply rounding (if enabled) and report the issued command."""
        q_cmd = self._issued_command(state.q_cmd)
        if self._cfg.quantization_rad <= 0:
            return RobotState(qpos=state.qpos, qvel=state.qvel, q_cmd=q_cmd, tcp_pos=state.tcp_pos)
        q = np.round(state.qpos / self._cfg.quantization_rad) * self._cfg.quantization_rad
        qvel = np.zeros_like(q) if self._prev_q is None else (q - self._prev_q) / self._dt
        self._prev_q = q
        return RobotState(qpos=q, qvel=qvel, q_cmd=q_cmd, tcp_pos=self._fk.forward(q))

    def _issued_command(self, inner_cmd: np.ndarray) -> np.ndarray:
        """Inner command plus deltas still in the queue, clipped to the limits."""
        if not self._queue:
            return inner_cmd.copy()
        limits = self.inner.joint_limits()
        cmd = inner_cmd.copy()
        n = len(limits.lower)
        cmd[:n] = np.clip(cmd[:n] + np.sum(self._queue, axis=0), limits.lower, limits.upper)
        return cmd

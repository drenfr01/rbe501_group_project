"""Typed value objects exchanged between environments, backends, and rewards."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Snapshot of the robot after a control step.

    Joint arrays are ordered arm joints first, then the gripper (radians, rad/s).
    `tcp_pos` is the world-frame gripper_site position in meters.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    q_cmd: np.ndarray
    tcp_pos: np.ndarray


@dataclass(frozen=True)
class JointLimits:
    """Lower and upper position limits (radians) for the arm joints."""

    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class RewardTerms:
    """Decomposed reach reward.

    `total = -distance_scale * distance + bonus - action_penalty - smoothness_penalty`.
    `distance` stays in meters so logs and eval stay in physical units.
    """

    distance: float
    bonus: float
    action_penalty: float
    smoothness_penalty: float
    total: float


@dataclass(frozen=True)
class ObservationLayout:
    """Named slices for the packed reach observation vector.

    Layout: `[q, q_dot, q_cmd - q, a_prev, p_tcp, p_target - p_tcp]`.
    """

    n_arm: int
    cartesian_dim: int = 3

    @property
    def qpos(self) -> slice:
        """Measured arm joint positions."""
        return slice(0, self.n_arm)

    @property
    def qvel(self) -> slice:
        """Measured arm joint velocities."""
        return slice(self.n_arm, 2 * self.n_arm)

    @property
    def cmd_err(self) -> slice:
        """Commanded minus measured arm joint positions."""
        return slice(2 * self.n_arm, 3 * self.n_arm)

    @property
    def prev_action(self) -> slice:
        """Action applied on the previous step."""
        return slice(3 * self.n_arm, 4 * self.n_arm)

    @property
    def tcp(self) -> slice:
        """World-frame gripper-site position."""
        start = 4 * self.n_arm
        return slice(start, start + self.cartesian_dim)

    @property
    def rel_target(self) -> slice:
        """Target minus gripper-site position."""
        start = 4 * self.n_arm + self.cartesian_dim
        return slice(start, start + self.cartesian_dim)

    @property
    def size(self) -> int:
        """Length of the packed observation vector."""
        return 4 * self.n_arm + 2 * self.cartesian_dim

    def pack(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        cmd_err: np.ndarray,
        prev_action: np.ndarray,
        tcp: np.ndarray,
        rel_target: np.ndarray,
    ) -> np.ndarray:
        """Concatenate state blocks into a float32 observation."""
        parts = (
            qpos[: self.n_arm],
            qvel[: self.n_arm],
            cmd_err[: self.n_arm],
            prev_action,
            tcp,
            rel_target,
        )
        return np.concatenate(parts).astype(np.float32)

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

    `total = -distance_scale * distance + bonus - action_penalty - smoothness_penalty
    - precision_penalty - velocity_penalty`.
    `distance` stays in meters so logs and eval stay in physical units.
    """

    distance: float
    bonus: float
    action_penalty: float
    smoothness_penalty: float
    total: float
    precision_penalty: float = 0.0
    velocity_penalty: float = 0.0


@dataclass(frozen=True)
class ObservationLayout:
    """Named slices for the packed reach observation vector.

    Layout: `[q, q_dot, q_cmd - q, a_prev, p_tcp, p_target - p_tcp]`, optionally
    followed by `fine_error` (clipped, scaled target error) and then
    `joint_goal` (IK joint angles for the target).
    """

    n_arm: int
    cartesian_dim: int = 3
    include_fine_error: bool = False
    include_joint_goal: bool = False

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
    def fine_error(self) -> slice | None:
        """Clipped, scaled target error, if included."""
        if not self.include_fine_error:
            return None
        start = self._base_size
        return slice(start, start + self.cartesian_dim)

    @property
    def joint_goal(self) -> slice | None:
        """IK joint goal for the arm, if included."""
        if not self.include_joint_goal:
            return None
        start = self._base_size + (self.cartesian_dim if self.include_fine_error else 0)
        return slice(start, start + self.n_arm)

    @property
    def size(self) -> int:
        """Length of the packed observation vector."""
        extra_fine = self.cartesian_dim if self.include_fine_error else 0
        extra_goal = self.n_arm if self.include_joint_goal else 0
        return self._base_size + extra_fine + extra_goal

    @property
    def _base_size(self) -> int:
        """Length of the always-present blocks."""
        return 4 * self.n_arm + 2 * self.cartesian_dim

    def pack(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        cmd_err: np.ndarray,
        prev_action: np.ndarray,
        tcp: np.ndarray,
        rel_target: np.ndarray,
        fine_error: np.ndarray | None = None,
        joint_goal: np.ndarray | None = None,
    ) -> np.ndarray:
        """Concatenate state blocks (and enabled extras) into a float32 observation."""
        n = self.n_arm
        parts = [qpos[:n], qvel[:n], cmd_err[:n], prev_action, tcp, rel_target]
        parts += self._extra("fine_error", self.include_fine_error, fine_error)
        parts += self._extra("joint_goal", self.include_joint_goal, joint_goal)
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _extra(name: str, enabled: bool, block: np.ndarray | None) -> list[np.ndarray]:
        """Return the block if enabled; fail if an enabled block is missing."""
        if not enabled:
            return []
        if block is None:
            raise ValueError(f"layout includes {name} but none was given")
        return [block]

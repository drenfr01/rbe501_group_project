"""Abstraction boundaries between the task, the robot, targets, and rewards."""

from typing import Protocol, runtime_checkable

import numpy as np

from rbe_501_so101.envs.types import JointLimits, RewardTerms, RobotState


@runtime_checkable
class RobotBackend(Protocol):
    """Interface for any SO-101 (simulated or physical) driven by joint commands."""

    def reset(self, rng: np.random.Generator) -> RobotState:
        """Return the robot to its home pose and report the resulting state."""
        ...

    def apply_delta(self, delta_arm_rad: np.ndarray) -> None:
        """Add a relative change to the commanded arm joint angles."""
        ...

    def step(self) -> RobotState:
        """Advance one control period under the current command."""
        ...

    def joint_limits(self) -> JointLimits:
        """Return the arm joint position limits."""
        ...

    def close(self) -> None:
        """Release all resources held by the backend."""
        ...


@runtime_checkable
class TargetMarker(Protocol):
    """Optional backend capability: visualize the current target."""

    def set_target_marker(self, xyz: np.ndarray) -> None:
        """Move the target marker to a world-frame XYZ position."""
        ...


@runtime_checkable
class Renderable(Protocol):
    """Optional backend capability: render the scene."""

    def render(self) -> np.ndarray | None:
        """Return an RGB frame (rgb_array mode) or update a viewer (human mode)."""
        ...

    def is_open(self) -> bool:
        """Return False once a human viewer window has been closed."""
        ...


class TargetSampler(Protocol):
    """Produces world-frame XYZ reach targets."""

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Return a target position in meters."""
        ...


class RewardFunction(Protocol):
    """Computes reward terms from the robot state, target, and action."""

    def __call__(
        self, state: RobotState, target: np.ndarray, action: np.ndarray
    ) -> RewardTerms:
        """Return the decomposed reward for one step."""
        ...

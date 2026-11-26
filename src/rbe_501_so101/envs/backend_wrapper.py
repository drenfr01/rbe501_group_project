"""Base class for `RobotBackend` decorators that forward everything they don't change."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from rbe_501_so101.envs.protocols import Renderable, RobotBackend, TargetMarker
from rbe_501_so101.envs.types import JointLimits, RobotState


class BackendWrapper(ABC):
    """Wraps a `RobotBackend`; subclasses override how commands and states pass through."""

    def __init__(self, inner: RobotBackend) -> None:
        self.inner = inner

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> RobotState:
        """Reset the inner backend and this wrapper's state."""

    @abstractmethod
    def apply_delta(self, delta_arm_rad: np.ndarray) -> None:
        """Take this step's command change."""

    @abstractmethod
    def step(self) -> RobotState:
        """Advance one control period."""

    @property
    def model(self) -> Any:
        """Inner MuJoCo model, if any (lets the env share it with samplers and IK)."""
        return getattr(self.inner, "model", None)

    @property
    def ids(self) -> Any:
        """Inner model indices, if any."""
        return getattr(self.inner, "ids", None)

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

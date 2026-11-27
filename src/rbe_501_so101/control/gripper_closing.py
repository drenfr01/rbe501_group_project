"""Opt-in command profiles for isolated closing experiments; no actuator edits."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClosingProfile:
    """Position-command speed/endpoint and optional privileged simulation-contact stop."""

    name: str
    target_deg: float = -10.
    rate_rad: float = .05
    stop_on_contact: bool = False
    stable_steps: int = 1

    def __post_init__(self) -> None:
        """Reject nonpositive speed and contact-count settings."""
        if self.rate_rad <= 0 or self.stable_steps < 1:
            raise ValueError("Positive closing rate and stable contact count required")


@dataclass
class ClosingLatch:
    """Freeze the command after sufficient two-pad contact; never rearm after loss."""

    profile: ClosingProfile
    count: int = 0
    latched: float | None = None

    def target(self, command: float, grasped: bool) -> float:
        """Return a position target, not a force or torque controller output."""
        if self.latched is not None:
            return self.latched
        self.count = self.count + 1 if grasped else 0
        if self.profile.stop_on_contact and self.count >= self.profile.stable_steps:
            self.latched = float(command)
            return self.latched
        return float(np.deg2rad(self.profile.target_deg))


def closing_action(command: float, target: float, profile: ClosingProfile, full_scale: float) -> float:
    """Normalize the rate-limited angle increment using the unchanged environment scale."""
    if full_scale <= 0:
        raise ValueError("Positive action scale required")
    limit = min(1., profile.rate_rad / full_scale)
    return float(np.clip((target - command) / full_scale, -limit, limit))

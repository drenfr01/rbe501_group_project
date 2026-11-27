"""Opt-in bounded Cartesian tracking correction for the final supported lowering stage."""

from dataclasses import dataclass

import numpy as np

from rbe_501_so101.control.carry_place import TransferConfig, TransferPhase
from rbe_501_so101.control.supported_place import SupportSearchConfig, SupportedCarryPlace


@dataclass(frozen=True)
class PlacementTrackingConfig:
    """Meter-valued integral compensation; gates, gains and support-search goal stay unchanged."""

    gain: float = .2
    correction_limit_m: float = .006
    activation_distance_m: float = .015
    reference_tolerance_m: float = .001

    def __post_init__(self) -> None:
        """Require a positive bounded compensator instead of an unrestricted target shift."""
        if not 0 < self.gain <= 1 or not 0 < self.correction_limit_m <= .006:
            raise ValueError("Positive gain <=1 and correction bound <=6mm required")
        if self.activation_distance_m <= 0 or self.reference_tolerance_m <= 0:
            raise ValueError("Positive activation and reference tolerances required")


class TrackedCarryPlace(SupportedCarryPlace):
    """Correct near-table tracking error without treating the numerical IK residual as reality."""

    def __init__(self, env, cfg: TransferConfig = TransferConfig(), support: SupportSearchConfig = SupportSearchConfig(),
                 tracking: PlacementTrackingConfig = PlacementTrackingConfig()) -> None:
        """Keep the two frozen prior controllers available and own only the new bias state."""
        super().__init__(env, cfg, support)
        self.tracking = tracking
        self.tracking_bias = np.zeros(3)

    def move_cube(self, goal: np.ndarray, phase: TransferPhase, budget: int) -> bool:
        """Start each lowering stage with zero correction; leave carry behavior unchanged."""
        if phase == TransferPhase.LOWER:
            self.tracking_bias[:] = 0
        return super().move_cube(goal, phase, budget)

    def update_tracking_bias(self, goal: np.ndarray) -> None:
        """Integrate only near the target after the current reference catches its requested goal."""
        if self.reference is None or self.support_ready():
            return
        cube_error = self.target - self.physical.grasp_state().obj.p_s
        caught_up = np.linalg.norm(self.reference - (goal + self.tracking_bias)) <= self.tracking.reference_tolerance_m
        if np.linalg.norm(cube_error) > self.tracking.activation_distance_m or not caught_up:
            return
        tracking_error = goal - self.backend.grasp_state().p_pinch
        limit = self.tracking.correction_limit_m
        self.tracking_bias = np.clip(self.tracking_bias + self.tracking.gain * tracking_error, -limit, limit)

    def move(self, goal: np.ndarray, phase: TransferPhase) -> bool:
        """Pass corrected references through the same rate, IK, clearance and contact checks."""
        if phase == TransferPhase.LOWER:
            self.update_tracking_bias(goal)
            goal = goal + self.tracking_bias
        return super().move(goal, phase)

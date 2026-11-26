"""Contact-aware final placement: stop downward references while correcting supported XY."""

import numpy as np

from rbe_501_so101.control.carry_place import TransferPhase
from rbe_501_so101.control.supported_place import SupportedCarryPlace
from rbe_501_so101.control.tracked_place import TrackedCarryPlace


class ContactTrackedCarryPlace(TrackedCarryPlace):
    """Retain the bounded airborne correction but avoid pressing into an already supporting table."""

    def update_tracking_bias(self, goal: np.ndarray) -> None:
        """Freeze vertical integration on support; allow only caught-up lateral correction."""
        if not self.contacts()[0]:
            return super().update_tracking_bias(goal)
        if self.reference is None or self.support_ready():
            return
        requested = goal + self.tracking_bias
        if np.linalg.norm((self.reference - requested)[:2]) > self.tracking.reference_tolerance_m:
            return
        error = (goal - self.backend.grasp_state().p_pinch)[:2]
        limit = self.tracking.correction_limit_m
        self.tracking_bias[:2] = np.clip(self.tracking_bias[:2] + self.tracking.gain * error, -limit, limit)

    def move(self, goal: np.ndarray, phase: TransferPhase) -> bool:
        """Hold the current vertical reference during support, keeping all inherited motion guards."""
        if phase != TransferPhase.LOWER:
            return super().move(goal, phase)
        self.update_tracking_bias(goal)
        corrected = goal + self.tracking_bias
        if self.contacts()[0] and self.reference is not None:
            corrected[2] = self.reference[2]
        return SupportedCarryPlace.move(self, corrected, phase)

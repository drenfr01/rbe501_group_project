"""Small bounded touchdown probe, retaining the complete-task release/contact guards."""

from dataclasses import dataclass

import numpy as np

from rbe_501_so101.control.carry_place import CarryPlace, TransferConfig, TransferPhase


@dataclass(frozen=True)
class SupportSearchConfig:
    """Maximum downward reference adjustment; actual table support ends the search."""

    maximum_descent_m: float = .003

    def __post_init__(self) -> None:
        """Reject unbounded or upward contact-search references."""
        if not 0 < self.maximum_descent_m <= .003:
            raise ValueError("Support search must be positive and bounded to 3 mm")


class SupportedCarryPlace(CarryPlace):
    """Seek measured table support instead of relying on exact resting-height tracking."""

    def __init__(self, env, cfg: TransferConfig = TransferConfig(), support: SupportSearchConfig = SupportSearchConfig()) -> None:
        """Keep the original transfer configuration and add one explicit probe bound."""
        super().__init__(env, cfg)
        self.support = support

    def move_cube(self, goal: np.ndarray, phase: TransferPhase, budget: int) -> bool:
        """Allow a small lower reference while the original stable-support gate stops motion."""
        adjusted = goal.copy()
        if phase == TransferPhase.LOWER:
            adjusted[2] -= self.support.maximum_descent_m
        return super().move_cube(adjusted, phase, budget)

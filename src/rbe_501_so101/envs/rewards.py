"""Reward functions for SO-101 tasks."""

import numpy as np

from rbe_501_so101.config import RewardConfig
from rbe_501_so101.envs.types import RewardTerms, RobotState


class ReachReward:
    """Dense reach reward with a centimeter-scale approach term.

    `total = -scale * d + w * exp(-(d / sigma)^2) - c * ||a||^2 - s * ||a - a_prev||^2`.

    The Gaussian bonus stays sharp (sigma = 5 mm) so a sagging arm is not paid.
    Scaling `d` to centimeters makes the last few centimeters of approach visible
    to the learner. The smoothness term penalizes action chatter while holding.
    """

    def __init__(self, cfg: RewardConfig = RewardConfig()) -> None:
        self._cfg = cfg

    def __call__(
        self,
        state: RobotState,
        target: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
    ) -> RewardTerms:
        """Return the decomposed reward for one step."""
        distance = float(np.linalg.norm(state.tcp_pos - target))
        bonus = self._bonus(distance)
        action = np.asarray(action, dtype=float)
        prev_action = np.asarray(prev_action, dtype=float)
        penalty = self._cfg.action_penalty * float(action @ action)
        delta = action - prev_action
        smoothness = self._cfg.smoothness_penalty * float(delta @ delta)
        total = -self._cfg.distance_scale * distance + bonus - penalty - smoothness
        return RewardTerms(distance, bonus, penalty, smoothness, total)

    def _bonus(self, distance: float) -> float:
        """Sharp Gaussian that is only meaningful inside about one centimeter."""
        sigma = self._cfg.bonus_sigma_m
        return self._cfg.bonus_weight * float(np.exp(-((distance / sigma) ** 2)))

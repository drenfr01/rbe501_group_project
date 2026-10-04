"""Reward functions for SO-101 tasks."""

import numpy as np

from rbe_501_so101.config import RewardConfig
from rbe_501_so101.envs.types import RewardTerms, RobotState


class ReachReward:
    """Dense reach reward: -d + w * exp(-(d / sigma)^2) - c * ||a||^2.

    The Gaussian bonus is sharp (sigma = 5 mm by default) so that closing the
    last centimeter of gravity sag is worth almost the full bonus.
    """

    def __init__(self, cfg: RewardConfig = RewardConfig()) -> None:
        self._cfg = cfg

    def __call__(
        self, state: RobotState, target: np.ndarray, action: np.ndarray
    ) -> RewardTerms:
        """Return the decomposed reward for one step."""
        distance = float(np.linalg.norm(state.tcp_pos - target))
        bonus = self._cfg.bonus_weight * float(np.exp(-((distance / self._cfg.bonus_sigma_m) ** 2)))
        action = np.asarray(action, dtype=float)
        penalty = self._cfg.action_penalty * float(action @ action)
        return RewardTerms(
            distance=distance,
            bonus=bonus,
            action_penalty=penalty,
            total=-distance + bonus - penalty,
        )

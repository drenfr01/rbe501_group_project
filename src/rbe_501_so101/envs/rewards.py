"""Reward functions for SO-101 tasks."""

import numpy as np

from rbe_501_so101.config import RewardConfig
from rbe_501_so101.envs.types import RewardTerms, RobotState


class ReachReward:
    """Dense reach reward with a centimeter-scale approach term.

    `total = -scale * d + w * exp(-(d / sigma)^2) - c * ||a||^2 - s * ||a - a_prev||^2
    - p * ln(1 + d / eps) - v * ||q_dot||^2 * exp(-(d / r)^2)`.

    The Gaussian bonus stays sharp (sigma = 5 mm) so a sagging arm is not paid.
    Scaling `d` to centimeters makes the last few centimeters of approach visible
    to the learner. The smoothness term penalizes action chatter while holding.
    The optional log term sharpens the gradient at millimeter range, and the
    optional velocity term asks the arm to be still once it is at the target.
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
        penalty = self._action_penalty(action)
        smoothness = self._smoothness_penalty(action, prev_action)
        precision = self._precision_penalty(distance)
        velocity = self._velocity_penalty(state.qvel, distance)
        approach = -self._cfg.distance_scale * distance
        total = approach + bonus - penalty - smoothness - precision - velocity
        return RewardTerms(
            distance=distance,
            bonus=bonus,
            action_penalty=penalty,
            smoothness_penalty=smoothness,
            total=total,
            precision_penalty=precision,
            velocity_penalty=velocity,
        )

    def _bonus(self, distance: float) -> float:
        """Sharp Gaussian that is only meaningful inside about one centimeter."""
        sigma = self._cfg.bonus_sigma_m
        return self._cfg.bonus_weight * float(np.exp(-((distance / sigma) ** 2)))

    def _action_penalty(self, action: np.ndarray) -> float:
        """Squared action magnitude."""
        action = np.asarray(action, dtype=float)
        return self._cfg.action_penalty * float(action @ action)

    def _smoothness_penalty(self, action: np.ndarray, prev_action: np.ndarray) -> float:
        """Squared change between consecutive actions."""
        delta = np.asarray(action, dtype=float) - np.asarray(prev_action, dtype=float)
        return self._cfg.smoothness_penalty * float(delta @ delta)

    def _precision_penalty(self, distance: float) -> float:
        """Log-distance term whose slope is largest within a few millimeters."""
        return self._cfg.precision_weight * float(np.log1p(distance / self._cfg.precision_eps_m))

    def _velocity_penalty(self, qvel: np.ndarray, distance: float) -> float:
        """Squared joint speed, weighted by a Gaussian of the target distance."""
        qvel = np.asarray(qvel, dtype=float)
        nearness = float(np.exp(-((distance / self._cfg.velocity_radius_m) ** 2)))
        return self._cfg.velocity_penalty * nearness * float(qvel @ qvel)

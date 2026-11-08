"""Reward functions for SO-101 tasks."""

import numpy as np

from rbe_501_so101.config import PickLiftRewardConfig, RewardConfig
from rbe_501_so101.envs.types import GraspState, PickLiftRewardTerms, RewardTerms, RobotState


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


class PickLiftReward:
    """Reward reach, two-pad grasp, and lift while penalizing action size."""

    def __init__(self, cfg: PickLiftRewardConfig = PickLiftRewardConfig(), lift_success_m: float = 0.04) -> None:
        self._cfg = cfg
        self._lift_success_m = lift_success_m

    def __call__(self, state: GraspState, action: np.ndarray) -> PickLiftRewardTerms:
        """Return the decomposed reward for one step."""
        distance = float(np.linalg.norm(state.obj.p_s - state.p_pinch))
        lift_m = state.obj.lift_m
        reach = 1.0 - float(np.tanh(self._cfg.reach_scale_per_m * distance))
        grasp = self._cfg.grasp_bonus if state.grasped else 0.0
        lift = self._lift_term(lift_m) if state.grasped else 0.0
        success = self._cfg.success_bonus if self.is_success(state) else 0.0
        action = np.asarray(action, dtype=float)
        penalty = self._cfg.action_penalty * float(action @ action)
        total = reach + grasp + lift + success - penalty
        return PickLiftRewardTerms(reach, grasp, lift, success, penalty, total, distance, lift_m)

    def is_success(self, state: GraspState) -> bool:
        """Object held by both pads at or above the success height."""
        return state.grasped and state.obj.lift_m >= self._lift_success_m

    def _lift_term(self, lift_m: float) -> float:
        """Linear in height up to the success height, then flat."""
        return self._cfg.lift_weight * float(np.clip(lift_m / self._lift_success_m, 0.0, 1.0))

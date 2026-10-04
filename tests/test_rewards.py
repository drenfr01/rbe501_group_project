"""Unit tests for ReachReward: bonus shape, monotonicity, and action penalty."""

import numpy as np
import pytest

from rbe_501_so101.config import RewardConfig
from rbe_501_so101.envs.types import RewardTerms, RobotState

SAG_DISTANCE_M = 0.012
NEGLIGIBLE_BONUS_FRACTION = 0.01
N_ARM = 5


def _state_at_distance(distance: float) -> tuple[RobotState, np.ndarray]:
    """Build a robot state whose TCP is `distance` meters from the returned target."""
    zeros = np.zeros(N_ARM + 1)
    target = np.array([0.1, 0.0, 0.6])
    tcp = target + np.array([0.0, 0.0, -distance])
    return RobotState(qpos=zeros, qvel=zeros, q_cmd=zeros, tcp_pos=tcp), target


@pytest.fixture
def reward_fn(reward_cfg):
    from rbe_501_so101.envs.rewards import ReachReward

    return ReachReward(reward_cfg)


def _terms(reward_fn, distance: float, action: np.ndarray | None = None) -> RewardTerms:
    state, target = _state_at_distance(distance)
    return reward_fn(state, target, np.zeros(N_ARM) if action is None else action)


def test_bonus_is_full_weight_at_zero_distance(reward_fn, reward_cfg):
    assert _terms(reward_fn, 0.0).bonus == pytest.approx(reward_cfg.bonus_weight)


def test_bonus_is_weight_over_e_at_sigma(reward_fn, reward_cfg):
    terms = _terms(reward_fn, reward_cfg.bonus_sigma_m)
    assert terms.bonus == pytest.approx(reward_cfg.bonus_weight * np.exp(-1.0))


def test_bonus_is_negligible_at_sag_distance(reward_fn, reward_cfg):
    terms = _terms(reward_fn, SAG_DISTANCE_M)
    assert terms.bonus < NEGLIGIBLE_BONUS_FRACTION * reward_cfg.bonus_weight


def test_reward_decreases_monotonically_with_distance(reward_fn):
    totals = [_terms(reward_fn, d).total for d in np.linspace(0.0, 0.3, 301)]
    assert np.all(np.diff(totals) < 0)


def test_action_penalty_matches_squared_norm(reward_fn, reward_cfg):
    action = np.array([0.5, -1.0, 0.25, 0.0, 1.0])
    terms = _terms(reward_fn, 0.0, action)
    assert terms.action_penalty == pytest.approx(reward_cfg.action_penalty * float(action @ action))


def test_total_combines_terms(reward_fn):
    terms = _terms(reward_fn, 0.004, np.full(N_ARM, 0.3))
    assert terms.distance == pytest.approx(0.004)
    assert terms.total == pytest.approx(-terms.distance + terms.bonus - terms.action_penalty)


def test_reward_uses_config_weights():
    from rbe_501_so101.envs.rewards import ReachReward

    cfg = RewardConfig(bonus_weight=2.0, bonus_sigma_m=0.01, action_penalty=0.0)
    terms = _terms(ReachReward(cfg), 0.0, np.ones(N_ARM))
    assert terms.bonus == pytest.approx(2.0)
    assert terms.action_penalty == 0.0

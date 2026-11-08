"""Unit tests for the staged pick-and-lift reward: reach, grasp, lift, success."""

import numpy as np
import pytest

from rbe_501_so101.config import PickLiftRewardConfig
from rbe_501_so101.envs.rewards import PickLiftReward
from rbe_501_so101.envs.types import GraspState, ObjectState, RobotState

N_JOINTS = 6
REST_Z = 0.49
LIFT_SUCCESS_M = 0.04
P_OBJ_REST = np.array([0.02, 0.0, REST_Z])


def _state(distance: float = 0.0, lift: float = 0.0, grasped: bool = False) -> GraspState:
    """Grasp state whose pinch point is `distance` above an object lifted by `lift` meters."""
    zeros = np.zeros(N_JOINTS)
    p_obj = P_OBJ_REST + np.array([0.0, 0.0, lift])
    robot = RobotState(qpos=zeros, qvel=zeros, q_cmd=zeros, tcp_pos=p_obj)
    obj = ObjectState(p_s=p_obj, rest_z=REST_Z, mass_kg=0.05)
    return GraspState(robot=robot, p_pinch=p_obj + np.array([0.0, 0.0, distance]), obj=obj, grasped=grasped)


@pytest.fixture
def cfg() -> PickLiftRewardConfig:
    """Default pick-and-lift reward weights."""
    return PickLiftRewardConfig()


@pytest.fixture
def reward_fn(cfg: PickLiftRewardConfig) -> PickLiftReward:
    """Reward with the default weights and a 4 cm success height."""
    return PickLiftReward(cfg, lift_success_m=LIFT_SUCCESS_M)


def _score(reward_fn: PickLiftReward, state: GraspState, action: np.ndarray | None = None):
    """Score a state with a zero (or given) action."""
    return reward_fn(state, np.zeros(N_JOINTS) if action is None else action)


def test_reach_term_is_one_at_zero_distance(reward_fn):
    assert _score(reward_fn, _state(0.0)).reach == pytest.approx(1.0)


def test_reach_term_decreases_with_distance(reward_fn):
    reach = [_score(reward_fn, _state(d)).reach for d in np.linspace(0.0, 0.3, 31)]
    assert np.all(np.diff(reach) < 0)


def test_no_grasp_means_no_grasp_or_lift_reward(reward_fn):
    terms = _score(reward_fn, _state(lift=0.05, grasped=False))
    assert terms.grasp == 0.0 and terms.lift == 0.0 and terms.success == 0.0


def test_grasp_bonus_paid_when_grasped(reward_fn, cfg):
    assert _score(reward_fn, _state(grasped=True)).grasp == pytest.approx(cfg.grasp_bonus)


def test_lift_term_grows_with_height_and_saturates(reward_fn, cfg):
    half = _score(reward_fn, _state(lift=LIFT_SUCCESS_M / 2, grasped=True)).lift
    full = _score(reward_fn, _state(lift=LIFT_SUCCESS_M, grasped=True)).lift
    above = _score(reward_fn, _state(lift=3 * LIFT_SUCCESS_M, grasped=True)).lift
    assert half == pytest.approx(cfg.lift_weight / 2)
    assert full == pytest.approx(cfg.lift_weight) and above == pytest.approx(cfg.lift_weight)


def test_success_requires_grasp_and_height(reward_fn, cfg):
    assert _score(reward_fn, _state(lift=LIFT_SUCCESS_M, grasped=True)).success == cfg.success_bonus
    assert _score(reward_fn, _state(lift=0.9 * LIFT_SUCCESS_M, grasped=True)).success == 0.0


def test_total_is_sum_of_terms_minus_action_penalty(reward_fn, cfg):
    action = np.full(N_JOINTS, 0.5)
    terms = _score(reward_fn, _state(distance=0.01, lift=0.02, grasped=True), action)
    expected = terms.reach + terms.grasp + terms.lift + terms.success - terms.action_penalty
    assert terms.action_penalty == pytest.approx(cfg.action_penalty * float(action @ action))
    assert terms.total == pytest.approx(expected)


def test_terms_report_physical_distance_and_lift(reward_fn):
    terms = _score(reward_fn, _state(distance=0.01, lift=0.02, grasped=True))
    assert terms.distance_m == pytest.approx(0.01)
    assert terms.lift_m == pytest.approx(0.02)

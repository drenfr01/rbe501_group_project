"""Unit tests for reach-env variants: shaped actions, fine error observation, residual IK."""

from collections.abc import Iterator

import numpy as np
import pytest

from rbe_501_so101.config import ReachTaskConfig, SimConfig
from rbe_501_so101.envs.actions import shape_action
from rbe_501_so101.envs.protocols import JointGoalSolver
from rbe_501_so101.envs.reach_env import SO101ReachEnv
from rbe_501_so101.envs.types import ObservationLayout

N_ARM = len(SimConfig().arm_joints)
SEED = 3
GOAL_OFFSET_RAD = np.array([0.3, -0.2, 0.25, 0.1, -0.15])
SETTLE_STEPS = 40


class FixedOffsetSolver:
    """Fake IK: the goal is always the start command plus a fixed offset."""

    def __init__(self) -> None:
        self.calls = 0

    def joint_goal(self, target: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """Return the arm part of `q_init` shifted by `GOAL_OFFSET_RAD`."""
        self.calls += 1
        return np.asarray(q_init[:N_ARM], dtype=float) + GOAL_OFFSET_RAD


@pytest.fixture
def cubic_env() -> Iterator[SO101ReachEnv]:
    """Delta-action env with a cubic action map."""
    env = SO101ReachEnv(task_cfg=ReachTaskConfig(action_exponent=3.0))
    yield env
    env.close()


@pytest.fixture
def residual_env() -> Iterator[tuple[SO101ReachEnv, FixedOffsetSolver]]:
    """Residual-IK env with a deterministic fake solver."""
    solver = FixedOffsetSolver()
    task = ReachTaskConfig(action_mode="residual_ik", fine_error_obs=True)
    env = SO101ReachEnv(task_cfg=task, goal_solver=solver)
    yield env, solver
    env.close()


def test_shape_action_preserves_sign_and_endpoints():
    action = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(shape_action(action, 3.0), [-1.0, -0.125, 0.0, 0.125, 1.0])
    np.testing.assert_allclose(shape_action(action, 1.0), action)


def test_cubic_exponent_shrinks_small_deltas(cubic_env):
    cubic_env.reset(seed=SEED)
    before = cubic_env.state.q_cmd.copy()
    action = np.array([0.5, -0.5, 0.5, -0.5, 0.5], dtype=np.float32)
    cubic_env.step(action)
    delta = cubic_env.state.q_cmd[:N_ARM] - before[:N_ARM]
    expected = np.sign(action) * 0.125 * ReachTaskConfig().delta_q_max_rad
    np.testing.assert_allclose(delta, expected, rtol=1e-5)


def test_default_layout_has_no_extra_blocks():
    layout = ObservationLayout(n_arm=N_ARM)
    assert layout.size == 4 * N_ARM + 6
    assert layout.fine_error is None and layout.joint_goal is None


def test_layout_appends_fine_error_then_joint_goal():
    layout = ObservationLayout(n_arm=N_ARM, include_fine_error=True, include_joint_goal=True)
    base = 4 * N_ARM + 6
    assert layout.fine_error == slice(base, base + 3)
    assert layout.joint_goal == slice(base + 3, base + 3 + N_ARM)
    assert layout.size == base + 3 + N_ARM


def test_pack_rejects_missing_extra_block():
    layout = ObservationLayout(n_arm=N_ARM, include_fine_error=True)
    zeros = np.zeros(N_ARM)
    with pytest.raises(ValueError, match="fine_error"):
        layout.pack(zeros, zeros, zeros, zeros, np.zeros(3), np.zeros(3))


def test_fine_error_is_clipped_and_scaled(residual_env):
    env, _ = residual_env
    obs, _ = env.reset(seed=SEED)
    clip = ReachTaskConfig().fine_error_clip_m
    expected = np.clip(env.target - env.state.tcp_pos, -clip, clip) / clip
    np.testing.assert_allclose(obs[env.layout.fine_error], expected, rtol=1e-5, atol=1e-6)
    assert env.observation_space.shape == (env.layout.size,)


def test_residual_reset_solves_ik_once_and_observes_goal(residual_env):
    env, solver = residual_env
    obs, _ = env.reset(seed=SEED)
    assert solver.calls == 1
    expected = env.state.q_cmd[:N_ARM] + GOAL_OFFSET_RAD
    np.testing.assert_allclose(obs[env.layout.joint_goal], expected, rtol=1e-6)


def test_residual_zero_action_moves_command_toward_goal_rate_limited(residual_env):
    env, _ = residual_env
    env.reset(seed=SEED)
    start = env.state.q_cmd[:N_ARM].copy()
    env.step(np.zeros(N_ARM, dtype=np.float32))
    step = env.state.q_cmd[:N_ARM] - start
    delta_max = ReachTaskConfig().delta_q_max_rad
    assert np.all(np.abs(step) <= delta_max + 1e-9)
    np.testing.assert_allclose(np.sign(step), np.sign(GOAL_OFFSET_RAD))


def test_residual_command_settles_at_goal_plus_scaled_residual(residual_env):
    env, _ = residual_env
    env.reset(seed=SEED)
    goal = env.state.q_cmd[:N_ARM] + GOAL_OFFSET_RAD
    action = np.array([1.0, -1.0, 0.5, 0.0, -0.5], dtype=np.float32)
    for _ in range(SETTLE_STEPS):
        env.step(action)
    expected = goal + ReachTaskConfig().residual_max_rad * action
    np.testing.assert_allclose(env.state.q_cmd[:N_ARM], expected, atol=1e-9)


def test_default_residual_env_builds_ik_solver():
    env = SO101ReachEnv(task_cfg=ReachTaskConfig(action_mode="residual_ik"))
    try:
        obs, _ = env.reset(seed=SEED)
        assert isinstance(env.goal_solver, JointGoalSolver)
        assert obs.shape == (env.layout.size,)
    finally:
        env.close()

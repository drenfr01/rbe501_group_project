"""Unit tests for SO101ReachEnv: spaces, observation layout, episode logic, registration."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from rbe_501_so101.config import DynamicsRandomizationConfig, ReachTaskConfig, SimConfig
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend
from rbe_501_so101.envs.reach_env import SO101ReachEnv
from rbe_501_so101.envs.types import ObservationLayout

N_ARM = len(SimConfig().arm_joints)
LAYOUT = ObservationLayout(n_arm=N_ARM)
SEED = 3
REWARD_KEYS = {
    "reward_distance",
    "reward_bonus",
    "reward_action_penalty",
    "reward_smoothness_penalty",
    "reward_total",
}


def test_action_space_is_normalized_5d_box(reach_env):
    space = reach_env.action_space
    assert isinstance(space, spaces.Box)
    assert space.shape == (N_ARM,) and space.dtype == np.float32
    assert np.all(space.low == -1.0) and np.all(space.high == 1.0)


def test_observation_space_matches_layout(reach_env):
    space = reach_env.observation_space
    assert isinstance(space, spaces.Box)
    assert space.shape == (LAYOUT.size,) and space.dtype == np.float32
    assert reach_env.layout == LAYOUT


def test_reset_and_step_signatures(reach_env):
    obs, info = reach_env.reset(seed=SEED)
    assert obs.shape == (LAYOUT.size,) and obs.dtype == np.float32
    assert isinstance(info, dict)
    result = reach_env.step(reach_env.action_space.sample())
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.dtype == np.float32 and isinstance(reward, float)
    assert terminated is False and truncated is False


def test_observation_layout_matches_backend_state(reach_env):
    reach_env.reset(seed=SEED)
    action = np.full(N_ARM, 0.5, dtype=np.float32)
    obs, *_ = reach_env.step(action)
    state, target, layout = reach_env.state, reach_env.target, reach_env.layout
    np.testing.assert_allclose(obs[layout.qpos], state.qpos[:N_ARM], rtol=1e-6)
    np.testing.assert_allclose(obs[layout.qvel], state.qvel[:N_ARM], rtol=1e-6)
    np.testing.assert_allclose(obs[layout.cmd_err], (state.q_cmd - state.qpos)[:N_ARM], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(obs[layout.prev_action], action)
    np.testing.assert_allclose(obs[layout.tcp], state.tcp_pos, rtol=1e-6)
    np.testing.assert_allclose(obs[layout.rel_target], target - state.tcp_pos, rtol=1e-5, atol=1e-7)


def test_reset_zeroes_previous_action(reach_env):
    reach_env.reset(seed=SEED)
    reach_env.step(np.ones(N_ARM, dtype=np.float32))
    obs, _ = reach_env.reset(seed=SEED)
    np.testing.assert_array_equal(obs[reach_env.layout.prev_action], 0.0)


def test_action_is_scaled_by_delta_max(reach_env):
    reach_env.reset(seed=SEED)
    before = reach_env.state.q_cmd.copy()
    reach_env.step(np.ones(N_ARM, dtype=np.float32))
    delta = reach_env.state.q_cmd[:N_ARM] - before[:N_ARM]
    np.testing.assert_allclose(delta, ReachTaskConfig().delta_q_max_rad, rtol=1e-6)


def test_out_of_range_action_is_clipped(reach_env):
    reach_env.reset(seed=SEED)
    before = reach_env.state.q_cmd.copy()
    reach_env.step(np.full(N_ARM, 5.0, dtype=np.float32))
    delta = reach_env.state.q_cmd[:N_ARM] - before[:N_ARM]
    np.testing.assert_allclose(delta, ReachTaskConfig().delta_q_max_rad, rtol=1e-6)


def test_gripper_is_held_at_configured_angle(reach_env):
    reach_env.reset(seed=SEED)
    reach_env.step(reach_env.action_space.sample())
    expected = np.deg2rad(ReachTaskConfig().gripper_hold_deg)
    np.testing.assert_allclose(reach_env.state.q_cmd[N_ARM], expected)


def test_episode_truncates_at_max_steps_and_never_terminates(reach_env):
    reach_env.reset(seed=SEED)
    max_steps = ReachTaskConfig().max_episode_steps
    flags = [reach_env.step(np.zeros(N_ARM, dtype=np.float32))[2:4] for _ in range(max_steps)]
    terminated, truncated = zip(*flags)
    assert not any(terminated)
    assert not any(truncated[:-1]) and truncated[-1]


def test_info_reports_distance_and_reward_terms(reach_env):
    reach_env.reset(seed=SEED)
    _, reward, _, _, info = reach_env.step(np.zeros(N_ARM, dtype=np.float32))
    expected_distance = np.linalg.norm(reach_env.state.tcp_pos - reach_env.target)
    assert info["distance_m"] == np.float64(expected_distance)
    assert REWARD_KEYS <= info.keys()
    assert info["reward_total"] == reward


def test_same_seed_gives_same_target(reach_env):
    reach_env.reset(seed=SEED)
    first = reach_env.target.copy()
    reach_env.reset(seed=SEED)
    np.testing.assert_array_equal(reach_env.target, first)


def test_target_marker_follows_target(reach_env):
    reach_env.reset(seed=SEED)
    backend = reach_env.backend
    assert isinstance(backend, MujocoSo101Backend)
    site = backend.model.site(SimConfig().target_site).id
    np.testing.assert_allclose(backend.data.site_xpos[site], reach_env.target, atol=1e-9)


def test_injected_backend_rejects_builder_kwargs(backend):
    with pytest.raises(ValueError, match="injecting a backend"):
        SO101ReachEnv(backend=backend, randomize_dynamics=True)
    with pytest.raises(ValueError, match="injecting a backend"):
        SO101ReachEnv(backend=backend, sim_cfg=SimConfig())
    with pytest.raises(ValueError, match="injecting a backend"):
        SO101ReachEnv(backend=backend, randomization_cfg=DynamicsRandomizationConfig())


def test_control_dt_matches_sim_config(reach_env):
    assert reach_env.control_dt == SimConfig().control_dt


def test_registered_env_id():
    import rbe_501_so101.envs  # noqa: F401

    env = gym.make("SO101Reach-v0")
    assert env.spec.max_episode_steps is None
    assert not isinstance(env, TimeLimit)
    obs, _ = env.reset(seed=SEED)
    assert obs.shape == (LAYOUT.size,)
    env.close()

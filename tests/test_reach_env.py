"""Unit tests for SO101ReachEnv: spaces, observation layout, episode logic, registration."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rbe_501_so101.config import ReachTaskConfig, SimConfig

N_ARM = len(SimConfig().arm_joints)
OBS_DIM = 26
SEED = 3

Q_SLICE = slice(0, 5)
QVEL_SLICE = slice(5, 10)
CMD_ERR_SLICE = slice(10, 15)
PREV_ACTION_SLICE = slice(15, 20)
TCP_SLICE = slice(20, 23)
REL_TARGET_SLICE = slice(23, 26)


def test_action_space_is_normalized_5d_box(reach_env):
    space = reach_env.action_space
    assert isinstance(space, spaces.Box)
    assert space.shape == (N_ARM,) and space.dtype == np.float32
    assert np.all(space.low == -1.0) and np.all(space.high == 1.0)


def test_observation_space_is_26d_float32_box(reach_env):
    space = reach_env.observation_space
    assert isinstance(space, spaces.Box)
    assert space.shape == (OBS_DIM,) and space.dtype == np.float32


def test_reset_and_step_signatures(reach_env):
    obs, info = reach_env.reset(seed=SEED)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
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
    state, target = reach_env.state, reach_env.target
    np.testing.assert_allclose(obs[Q_SLICE], state.qpos[:N_ARM], rtol=1e-6)
    np.testing.assert_allclose(obs[QVEL_SLICE], state.qvel[:N_ARM], rtol=1e-6)
    np.testing.assert_allclose(obs[CMD_ERR_SLICE], (state.q_cmd - state.qpos)[:N_ARM], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(obs[PREV_ACTION_SLICE], action)
    np.testing.assert_allclose(obs[TCP_SLICE], state.tcp_pos, rtol=1e-6)
    np.testing.assert_allclose(obs[REL_TARGET_SLICE], target - state.tcp_pos, rtol=1e-5, atol=1e-7)


def test_reset_zeroes_previous_action(reach_env):
    reach_env.reset(seed=SEED)
    reach_env.step(np.ones(N_ARM, dtype=np.float32))
    obs, _ = reach_env.reset(seed=SEED)
    np.testing.assert_array_equal(obs[PREV_ACTION_SLICE], 0.0)


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
    assert {"reward_distance", "reward_bonus", "reward_action_penalty", "reward_total"} <= info.keys()
    assert info["reward_total"] == reward


def test_same_seed_gives_same_target(reach_env):
    reach_env.reset(seed=SEED)
    first = reach_env.target.copy()
    reach_env.reset(seed=SEED)
    np.testing.assert_array_equal(reach_env.target, first)


def test_target_marker_follows_target(reach_env):
    reach_env.reset(seed=SEED)
    backend = reach_env.backend
    site = backend.model.site(SimConfig().target_site).id
    np.testing.assert_allclose(backend.data.site_xpos[site], reach_env.target, atol=1e-9)


def test_registered_env_id():
    import rbe_501_so101.envs  # noqa: F401

    env = gym.make("SO101Reach-v0")
    assert env.spec.max_episode_steps == ReachTaskConfig().max_episode_steps
    obs, _ = env.reset(seed=SEED)
    assert obs.shape == (OBS_DIM,)
    env.close()

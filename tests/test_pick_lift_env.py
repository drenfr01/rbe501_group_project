"""Unit tests for SO101PickLiftEnv: spaces, object placement, mass parameter, episode logic."""

import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from rbe_501_so101.config import ObjectMassConfig, ObjectPlacementConfig, PickLiftTaskConfig, SimConfig
from rbe_501_so101.envs import PICK_LIFT_ENV_ID, SO101PickLiftEnv

N_JOINTS = len(SimConfig().joints)
N_ARM = len(SimConfig().arm_joints)
SEED = 5
SETTLE_STEPS = 30
REST_DRIFT_TOL_M = 0.001
INFO_KEYS = {"mass_kg", "lift_m", "grasped", "is_success", "distance_m", "reward_total"}


@pytest.fixture
def pick_lift_env():
    """Headless pick-and-lift env used by this test module."""
    env = SO101PickLiftEnv()
    yield env
    env.close()


def test_action_space_is_normalized_6d_box(pick_lift_env):
    space = pick_lift_env.action_space
    assert isinstance(space, spaces.Box) and space.shape == (N_JOINTS,)
    assert np.all(space.low == -1.0) and np.all(space.high == 1.0)


def test_observation_matches_layout(pick_lift_env):
    obs, _ = pick_lift_env.reset(seed=SEED)
    layout = pick_lift_env.layout
    assert obs.shape == (layout.size,) == pick_lift_env.observation_space.shape
    np.testing.assert_allclose(obs[layout.object_pos], pick_lift_env.grasp_state.obj.p_s, rtol=1e-6)
    expected = pick_lift_env.grasp_state.obj.p_s - pick_lift_env.grasp_state.p_pinch
    np.testing.assert_allclose(obs[layout.object_error], expected, rtol=1e-5, atol=1e-7)


def test_object_spawns_resting_inside_placement_band(pick_lift_env):
    cfg = ObjectPlacementConfig()
    for seed in range(10):
        pick_lift_env.reset(seed=seed)
        obj = pick_lift_env.grasp_state.obj
        assert obj.p_s[2] == obj.rest_z
        radial = np.linalg.norm(obj.p_s[:2] - pick_lift_env.backend.pan_axis_xy)
        assert cfg.radial_min_m <= radial <= cfg.radial_max_m


def test_object_stays_at_rest_without_contact(pick_lift_env):
    pick_lift_env.reset(seed=SEED)
    start = pick_lift_env.grasp_state.obj.p_s.copy()
    for _ in range(SETTLE_STEPS):
        pick_lift_env.step(np.zeros(N_JOINTS, dtype=np.float32))
    assert np.linalg.norm(pick_lift_env.grasp_state.obj.p_s - start) < REST_DRIFT_TOL_M


def test_fixed_mass_is_applied_every_episode():
    env = SO101PickLiftEnv(mass_cfg=ObjectMassConfig(mode="fixed", fixed_kg=0.2))
    masses = {env.reset(seed=s)[1]["mass_kg"] for s in range(5)}
    assert masses == {0.2}
    assert env.backend.object_mass_kg == 0.2
    env.close()


def test_random_mass_varies_per_episode_within_range():
    cfg = ObjectMassConfig(mode="random", range_kg=(0.02, 0.6))
    env = SO101PickLiftEnv(mass_cfg=cfg)
    masses = [env.reset(seed=s)[1]["mass_kg"] for s in range(10)]
    assert len(set(masses)) == len(masses)
    assert all(0.02 <= m <= 0.6 for m in masses)
    env.close()


def test_actions_scale_arm_and_gripper_commands(pick_lift_env):
    task = PickLiftTaskConfig()
    pick_lift_env.reset(seed=SEED)
    before = pick_lift_env.grasp_state.robot.q_cmd.copy()
    pick_lift_env.step(np.full(N_JOINTS, -1.0, dtype=np.float32))
    delta = pick_lift_env.grasp_state.robot.q_cmd - before
    np.testing.assert_allclose(delta[:N_ARM], -task.delta_q_max_rad, rtol=1e-6)
    np.testing.assert_allclose(delta[N_ARM], -task.delta_gripper_max_rad, rtol=1e-6)


def test_episode_truncates_at_max_steps_and_never_terminates(pick_lift_env):
    pick_lift_env.reset(seed=SEED)
    steps = PickLiftTaskConfig().max_episode_steps
    flags = [pick_lift_env.step(np.zeros(N_JOINTS, dtype=np.float32))[2:4] for _ in range(steps)]
    terminated, truncated = zip(*flags, strict=True)
    assert not any(terminated) and not any(truncated[:-1]) and truncated[-1]


def test_info_reports_mass_lift_and_success(pick_lift_env):
    pick_lift_env.reset(seed=SEED)
    _, reward, _, _, info = pick_lift_env.step(np.zeros(N_JOINTS, dtype=np.float32))
    assert INFO_KEYS <= info.keys()
    assert info["reward_total"] == reward
    assert info["is_success"] is False and info["grasped"] is False


def test_check_env_passes_without_warnings(pick_lift_env):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(pick_lift_env, skip_render_check=True)


def test_registered_env_accepts_mass_config():
    env = gym.make(PICK_LIFT_ENV_ID, mass_cfg=ObjectMassConfig(mode="fixed", fixed_kg=0.3))
    _, info = env.reset(seed=SEED)
    assert info["mass_kg"] == 0.3
    env.close()

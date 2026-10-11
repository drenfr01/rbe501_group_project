"""Stability tests: Gymnasium conformance and long random rollouts without NaN or warnings."""

import warnings

import numpy as np
from gymnasium.utils.env_checker import check_env

from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend

N_RANDOM_STEPS = 1_000
SEED = 11


def test_check_env_passes_without_warnings(reach_env):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(reach_env.unwrapped, skip_render_check=True)


def test_random_rollout_is_finite_and_warning_free(reach_env):
    reach_env.action_space.seed(SEED)
    obs, _ = reach_env.reset(seed=SEED)
    assert isinstance(reach_env.backend, MujocoSo101Backend)
    for _ in range(N_RANDOM_STEPS):
        obs, reward, terminated, truncated, _ = reach_env.step(reach_env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        assert not reach_env.backend.has_warnings()
        if terminated or truncated:
            obs, _ = reach_env.reset()

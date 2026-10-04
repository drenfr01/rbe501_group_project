"""Shared pytest fixtures: configs, a seeded RNG, a MuJoCo backend, and a reach env."""

from collections.abc import Iterator

import mujoco
import numpy as np
import pytest

from rbe_501_so101.config import (
    DynamicsRandomizationConfig,
    ReachTaskConfig,
    RewardConfig,
    SimConfig,
    TargetSamplerConfig,
)

RNG_SEED = 1234


@pytest.fixture
def sim_cfg() -> SimConfig:
    """Default simulation configuration."""
    return SimConfig()


@pytest.fixture
def task_cfg() -> ReachTaskConfig:
    """Default reach task configuration."""
    return ReachTaskConfig()


@pytest.fixture
def sampler_cfg() -> TargetSamplerConfig:
    """Default target sampler configuration."""
    return TargetSamplerConfig()


@pytest.fixture
def reward_cfg() -> RewardConfig:
    """Default reward configuration."""
    return RewardConfig()


@pytest.fixture
def randomization_cfg() -> DynamicsRandomizationConfig:
    """Dynamics randomization configuration with randomization enabled."""
    return DynamicsRandomizationConfig(enabled=True)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def raw_model(sim_cfg: SimConfig) -> mujoco.MjModel:
    """The MJCF model exactly as written on disk (no runtime overrides)."""
    return mujoco.MjModel.from_xml_path(str(sim_cfg.model_path))


@pytest.fixture
def backend(sim_cfg: SimConfig) -> Iterator["MujocoSo101Backend"]:  # noqa: F821
    """A headless MuJoCo backend, closed after the test."""
    from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend

    instance = MujocoSo101Backend(sim_cfg)
    yield instance
    instance.close()


@pytest.fixture
def reach_env() -> Iterator["SO101ReachEnv"]:  # noqa: F821
    """A headless reach environment with default components, closed after the test."""
    from rbe_501_so101.envs.reach_env import SO101ReachEnv

    env = SO101ReachEnv()
    yield env
    env.close()

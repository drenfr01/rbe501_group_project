"""Gymnasium environments, backends, rewards, and target samplers for the SO-101."""

from gymnasium.envs.registration import register

from rbe_501_so101.config import ReachTaskConfig
from rbe_501_so101.envs.reach_env import SO101ReachEnv

REACH_ENV_ID = "SO101Reach-v0"

register(
    id=REACH_ENV_ID,
    entry_point="rbe_501_so101.envs.reach_env:SO101ReachEnv",
    max_episode_steps=ReachTaskConfig().max_episode_steps,
)

__all__ = ["REACH_ENV_ID", "SO101ReachEnv"]

"""Gymnasium environments, backends, rewards, and target samplers for the SO-101."""

from gymnasium.envs.registration import register

from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from rbe_501_so101.envs.reach_env import SO101ReachEnv

REACH_ENV_ID = "SO101Reach-v0"
PICK_LIFT_ENV_ID = "SO101PickLift-v0"
PICK_PLACE_ENV_ID = "SO101PickPlace-v1"

register(
    id=REACH_ENV_ID,
    entry_point="rbe_501_so101.envs.reach_env:SO101ReachEnv",
)
register(
    id=PICK_LIFT_ENV_ID,
    entry_point="rbe_501_so101.envs.pick_lift_env:SO101PickLiftEnv",
)

register(id=PICK_PLACE_ENV_ID, entry_point="rbe_501_so101.envs.pick_place_env:SO101PickPlaceEnv")

__all__ = ["PICK_PLACE_ENV_ID", "PICK_LIFT_ENV_ID", "REACH_ENV_ID", "SO101PickLiftEnv", "SO101ReachEnv"]

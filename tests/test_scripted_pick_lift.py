"""Physics tests: the scripted IK controller lifts a light cube; heavier cubes sag lower."""

import numpy as np
import pytest

from rbe_501_so101.config import ObjectMassConfig
from rbe_501_so101.control import ScriptedPickLift
from rbe_501_so101.envs import SO101PickLiftEnv

LIGHT_KG = 0.05
HEAVY_KG = 1.5
STALLING_KG = 3.0
SAG_MARGIN_M = 0.01


def _final_info(env: SO101PickLiftEnv, seed: int) -> dict:
    """Run one scripted episode and return the last step's info."""
    controller = ScriptedPickLift.for_env(env)
    obs, _ = env.reset(seed=seed)
    controller.reset()
    truncated, info = False, {}
    while not truncated:
        obs, _, _, truncated, info = env.step(controller(obs))
    return info


def _env(mass_kg: float) -> SO101PickLiftEnv:
    """Pick-and-lift env with a fixed object mass."""
    return SO101PickLiftEnv(mass_cfg=ObjectMassConfig(mode="fixed", fixed_kg=mass_kg))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scripted_controller_lifts_light_cube(seed):
    env = _env(LIGHT_KG)
    info = _final_info(env, seed)
    env.close()
    assert info["grasped"] and info["is_success"]


def test_heavier_cube_is_held_but_lifted_less():
    light_env, heavy_env = _env(LIGHT_KG), _env(HEAVY_KG)
    light, heavy = _final_info(light_env, seed=0), _final_info(heavy_env, seed=0)
    light_env.close()
    heavy_env.close()
    assert heavy["grasped"]
    assert heavy["lift_m"] < light["lift_m"] - SAG_MARGIN_M


def test_cube_beyond_stall_torque_is_not_lifted():
    env = _env(STALLING_KG)
    info = _final_info(env, seed=0)
    env.close()
    assert not info["is_success"]


def test_controller_actions_are_normalized():
    env = _env(LIGHT_KG)
    controller = ScriptedPickLift.for_env(env)
    obs, _ = env.reset(seed=0)
    controller.reset()
    actions = [controller(obs) for _ in range(5)]
    env.close()
    assert all(a.shape == env.action_space.shape and np.all(np.abs(a) <= 1.0) for a in actions)

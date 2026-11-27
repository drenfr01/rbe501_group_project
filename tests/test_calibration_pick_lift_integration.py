"""Calibration must also work with the runtime-compiled pick-and-lift model."""

from dataclasses import replace

import numpy as np
import pytest

from rbe_501_so101.config import DEFAULT_CALIBRATION_FILE, PickLiftConfig, resolve_project_path
from rbe_501_so101.envs.calibration import ActuatorCalibration
from rbe_501_so101.envs.grasp_model import FIXED_PAD_GEOM, MOVING_PAD_GEOM
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv


def test_pick_lift_applies_opt_in_stiffness_without_losing_grasp_geometry():
    """A compiled grasp model keeps its pads/cube when calibrated, including on reset."""
    config = PickLiftConfig()
    calibration_path = resolve_project_path(DEFAULT_CALIBRATION_FILE)
    config = replace(config, sim=replace(config.sim, calibration_path=calibration_path))
    calibration = ActuatorCalibration.load(calibration_path)
    env = SO101PickLiftEnv(cfg=config)
    try:
        for seed in (0, 1):
            env.reset(seed=seed)
            model = env.backend.model
            for entry in calibration.joints:
                actuator = model.actuator(entry.joint + config.sim.actuator_suffix).id
                assert model.actuator_gainprm[actuator, 0] == pytest.approx(entry.kp_nm_per_rad)
            for name in (FIXED_PAD_GEOM, MOVING_PAD_GEOM):
                assert model.geom(name).contype[0] != 0
            cube = model.geom(config.object.geom)
            np.testing.assert_allclose(cube.size, config.object.side_m / 2)
            assert model.body(config.object.body).mass[0] == pytest.approx(config.mass.fixed_kg)
    finally:
        env.close()

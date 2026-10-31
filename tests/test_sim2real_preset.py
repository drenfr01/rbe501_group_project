"""The sim2real training preset, the hardware speed cap, and the deployment condition."""

import json
from math import radians

import numpy as np
import pytest

from rbe_501_so101.config import (
    DEFAULT_CALIBRATION_FILE,
    PRESETS,
    ExperimentPreset,
    HardwareConfig,
    ReachTaskConfig,
)
from rbe_501_so101.envs.factory import make_reach_env
from rbe_501_so101.envs.perturbed_backend import PerturbedBackend
from rbe_501_so101.envs.servo_effects import ServoEffectsBackend
from rbe_501_so101.rl.acceptance import deployment_options
from rbe_501_so101.rl.robustness import hardware_speed_scale, slowed_preset

HW = HardwareConfig()
SIM2REAL = PRESETS["residual_ik_sim2real"]


def test_hardware_step_is_the_speed_scale_of_the_default_rate():
    """The real arm's command step is the default sim rate times `speed_scale`."""
    assert HW.max_command_step_rad == pytest.approx(HW.speed_scale * ReachTaskConfig().delta_q_max_rad)


@pytest.mark.parametrize(("delta", "scale"), [(0.07, 0.5), (0.035, 1.0), (0.02, 1.0)])
def test_hardware_speed_scale_caps_but_never_speeds_up(delta, scale):
    """Presets faster than the hardware are slowed to it; slower ones run as trained."""
    assert hardware_speed_scale(ReachTaskConfig(delta_q_max_rad=delta), HW) == pytest.approx(scale)


def test_sim2real_preset_trains_at_hardware_speed():
    """Deploying the new preset does not change its command rate."""
    assert SIM2REAL.task.delta_q_max_rad == pytest.approx(HW.max_command_step_rad)
    capped = slowed_preset(SIM2REAL, hardware_speed_scale(SIM2REAL.task, HW))
    assert capped.task.delta_q_max_rad == pytest.approx(SIM2REAL.task.delta_q_max_rad)


def test_sim2real_preset_models_the_measured_servo():
    """Calibration, stiction (randomized), LeRobot's clip, the real joint window, and a larger residual."""
    env = SIM2REAL.env
    assert env.calibration_path == DEFAULT_CALIBRATION_FILE
    assert env.servo.stiction
    low, high = env.servo.stiction_scale_range
    assert low < 1.0 < high
    assert env.servo.relative_clip_rad == pytest.approx(radians(HW.max_relative_target_deg))
    assert env.target_joint_window_rad == HW.sim_joint_window_rad()
    assert env.randomization.enabled
    assert env.perturbation.quantization_rad > 0.0 and env.perturbation.randomize_delay
    assert SIM2REAL.task.residual_max_rad > PRESETS["residual_ik"].task.residual_max_rad


def test_sim2real_preset_round_trips_through_json():
    """Saved run configs load back to the same preset."""
    data = json.loads(json.dumps(SIM2REAL.to_dict()))
    assert ExperimentPreset.from_dict(data) == SIM2REAL


def test_sim2real_env_builds_with_servo_effects_and_steps(rng):
    """The training env wraps MuJoCo in servo effects inside the perturbation, and runs."""
    env = make_reach_env(SIM2REAL, randomize=True)
    try:
        assert isinstance(env.backend, PerturbedBackend)
        assert isinstance(env.backend.inner, ServoEffectsBackend)
        obs, _ = env.reset(seed=0)
        for _ in range(5):
            obs, reward, *_ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
    finally:
        env.close()


def test_deployment_options_fix_every_random_draw_at_the_measured_value():
    """Deployment condition: calibrated arm, nominal stiction, the longest delay, no randomization."""
    options = deployment_options(SIM2REAL.env)
    assert not options.randomization.enabled
    assert options.servo.stiction_scale_range == (1.0, 1.0)
    assert options.perturbation.delay_steps == SIM2REAL.env.perturbation.delay_steps
    assert not options.perturbation.randomize_delay
    assert options.target_joint_window_rad == SIM2REAL.env.target_joint_window_rad


def test_sim2real_env_clips_commands_to_the_real_joint_window():
    """The command limits the policy meets in training are the real backend's."""
    env = make_reach_env(SIM2REAL, randomize=False)
    try:
        limits = env.backend.joint_limits()
        window = np.array(SIM2REAL.env.target_joint_window_rad)
        assert np.all(limits.lower >= window[:, 0] - 1e-12) and np.all(limits.upper <= window[:, 1] + 1e-12)
    finally:
        env.close()

"""Reach targets restricted to the joint range the real arm can reach (D-02)."""

import numpy as np
import pytest

from rbe_501_so101.config import HardwareConfig, SimConfig, TargetSamplerConfig
from rbe_501_so101.envs.targets import ReachableTargetSampler
from rbe_501_so101.hardware.joint_mapping import JointMap

SIM = SimConfig()
LIFT = 1


def test_window_matches_the_real_backend_limits(raw_model):
    """Same numbers the real backend clips to: calibrated real range in sim radians, minus the margin."""
    hw = HardwareConfig()
    joint_map = JointMap.from_model(raw_model, SIM, hw)
    limits_deg = {j: (-h, h) for j, h in zip(SIM.arm_joints, hw.real_half_range_deg, strict=True)}
    real = joint_map.arm_limits_to_sim(limits_deg)
    window = np.array(hw.sim_joint_window_rad())
    np.testing.assert_allclose(window[:, 0], real.lower + hw.limit_margin_rad)
    np.testing.assert_allclose(window[:, 1], real.upper - hw.limit_margin_rad)
    assert np.rad2deg(window[LIFT]) == pytest.approx((-106.0, 52.4), abs=0.5)


def test_sampler_bounds_are_the_tighter_of_model_and_window(raw_model):
    """The window only ever narrows the model range minus its margin."""
    window = HardwareConfig().sim_joint_window_rad()
    sampler = ReachableTargetSampler.from_config(SIM, TargetSamplerConfig(joint_window_rad=window))
    plain = ReachableTargetSampler.from_config(SIM, TargetSamplerConfig())
    low, high = sampler.joint_bounds
    plain_low, plain_high = plain.joint_bounds
    np.testing.assert_allclose(low, np.maximum(plain_low, [w[0] for w in window]))
    np.testing.assert_allclose(high, np.minimum(plain_high, [w[1] for w in window]))
    assert high[LIFT] < plain_high[LIFT]


def test_windowed_sampler_still_finds_targets(rng):
    """Restricting the range leaves plenty of valid targets."""
    cfg = TargetSamplerConfig(joint_window_rad=HardwareConfig().sim_joint_window_rad())
    sampler = ReachableTargetSampler.from_config(SIM, cfg)
    assert sampler.reachability_map(rng, 400).acceptance_rate > 0.1
    assert sampler.sample(rng).shape == (3,)


def test_window_reflects_the_lift_offset():
    """The −26.8° lift offset moves the real ±87.8° range to about −115..61 before the margin."""
    hw = HardwareConfig()
    low, high = np.rad2deg(hw.sim_joint_window_rad()[LIFT])
    margin = np.rad2deg(hw.limit_margin_rad)
    assert (low - margin, high + margin) == pytest.approx((-114.6, 61.0), abs=0.5)

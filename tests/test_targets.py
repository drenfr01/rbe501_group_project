"""Unit tests for ReachableTargetSampler: clearance, radial band, reachability, determinism."""

import mujoco
import numpy as np
import pytest

from rbe_501_so101.config import IKConfig, TargetSamplerConfig

N_BULK_SAMPLES = 1_000
N_IK_SAMPLES = 50
N_MAP_SAMPLES = 2_000


@pytest.fixture
def sampler(sim_cfg, sampler_cfg):
    from rbe_501_so101.envs.targets import ReachableTargetSampler

    return ReachableTargetSampler.from_config(sim_cfg, sampler_cfg)


@pytest.fixture
def pan_axis_xy(raw_model, sim_cfg) -> np.ndarray:
    """World XY of the shoulder-pan axis, computed independently of the sampler."""
    data = mujoco.MjData(raw_model)
    mujoco.mj_kinematics(raw_model, data)
    return data.xanchor[raw_model.joint(sim_cfg.radial_axis_joint).id][:2].copy()


def _radial(points: np.ndarray, axis_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, :2] - axis_xy, axis=1)


def test_targets_satisfy_clearance_and_radial_band(sampler, rng, sampler_cfg, pan_axis_xy):
    points = np.array([sampler.sample(rng) for _ in range(N_BULK_SAMPLES)])
    radial = _radial(points, pan_axis_xy)
    assert points.shape == (N_BULK_SAMPLES, 3)
    assert np.all(points[:, 2] >= sampler_cfg.z_min_m)
    assert np.all((radial >= sampler_cfg.radial_min_m) & (radial <= sampler_cfg.radial_max_m))


def test_every_target_is_reachable_by_ik(sampler, rng, sim_cfg):
    from rbe_501_so101.kinematics.ik import PositionIKSolver

    solver = PositionIKSolver.from_config(sim_cfg, IKConfig())
    home = np.deg2rad(sim_cfg.initial_deg)
    for _ in range(N_IK_SAMPLES):
        target = sampler.sample(rng)
        result = solver.solve(target, home)
        assert result.error_m < IKConfig().tolerance_m


def test_same_seed_produces_same_targets(sampler):
    first = [sampler.sample(np.random.default_rng(7)) for _ in range(3)]
    second = [sampler.sample(np.random.default_rng(7)) for _ in range(3)]
    np.testing.assert_array_equal(np.array(first), np.array(second))


def test_sampler_raises_when_band_is_empty(sim_cfg):
    from rbe_501_so101.envs.targets import ReachableTargetSampler

    impossible = TargetSamplerConfig(radial_min_m=5.0, radial_max_m=6.0, max_attempts=50)
    sampler = ReachableTargetSampler.from_config(sim_cfg, impossible)
    with pytest.raises(RuntimeError):
        sampler.sample(np.random.default_rng(0))


def test_reachability_map_flags_match_filters(sampler, rng, sampler_cfg, pan_axis_xy):
    reach_map = sampler.reachability_map(rng, N_MAP_SAMPLES)
    assert reach_map.points.shape == (N_MAP_SAMPLES, 3)
    np.testing.assert_allclose(reach_map.radial_m, _radial(reach_map.points, pan_axis_xy))
    expected = (
        (reach_map.points[:, 2] >= sampler_cfg.z_min_m)
        & (reach_map.radial_m >= sampler_cfg.radial_min_m)
        & (reach_map.radial_m <= sampler_cfg.radial_max_m)
    )
    np.testing.assert_array_equal(reach_map.accepted, expected)
    assert 0.0 < reach_map.acceptance_rate <= 1.0


def test_reachability_map_saves_npz(sampler, rng, tmp_path):
    path = tmp_path / "reach_map.npz"
    sampler.reachability_map(rng, 100).save(path)
    loaded = np.load(path)
    assert set(loaded.files) >= {"points", "radial_m", "accepted"}


def test_home_tcp_lies_inside_default_band(sampler, sim_cfg):
    from rbe_501_so101.kinematics.ik import PositionIKSolver

    solver = PositionIKSolver.from_config(sim_cfg, IKConfig())
    tcp = solver.forward(np.deg2rad(sim_cfg.initial_deg))
    radial = sampler.radial_distance(tcp)
    cfg = TargetSamplerConfig()
    assert cfg.radial_min_m <= radial <= cfg.radial_max_m

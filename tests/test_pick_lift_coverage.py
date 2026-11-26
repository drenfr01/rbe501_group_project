"""Diagnostic placement changes must preserve baseline sampling and observations."""

import numpy as np

from scripts.sim2real.analyze.pick_lift_coverage import CoverageProtocol, DiagnosticPlacer, PoseCase, cases


def test_zero_offset_preserves_original_rng_sequence() -> None:
    """An unrotated diagnostic wrapper must sample the identical baseline pose."""
    from rbe_501_so101.config import ObjectPlacementConfig
    from rbe_501_so101.envs.placement import TablePlacer

    base = TablePlacer(ObjectPlacementConfig(), np.array([.1, -.2]), .4)
    wrapped = DiagnosticPlacer(base, np.array([.1, -.2]), .4, PoseCase("random"))
    a, b = np.random.default_rng(82000), np.random.default_rng(82000)
    p, yaw = base.sample(a)
    actual, angle = wrapped.sample(b)
    np.testing.assert_array_equal(actual, p)
    assert angle == yaw
    assert a.random() == b.random()


def test_yaw_offset_changes_only_angle() -> None:
    """Rotated pairs keep their sampled center and random stream unchanged."""
    from rbe_501_so101.config import ObjectPlacementConfig
    from rbe_501_so101.envs.placement import TablePlacer

    base = TablePlacer(ObjectPlacementConfig(), np.zeros(2), .4)
    p, yaw = base.sample(np.random.default_rng(82000))
    wrapped = DiagnosticPlacer(base, np.zeros(2), .4, PoseCase("yaw", yaw_offset_deg=30))
    actual, angle = wrapped.sample(np.random.default_rng(82000))
    np.testing.assert_array_equal(actual, p)
    np.testing.assert_allclose(angle - yaw, np.deg2rad(30))


def test_coverage_matrix_stays_in_supported_workspace() -> None:
    """Yaw variations are labelled separately from in-domain workspace coverage."""
    matrix = cases(CoverageProtocol())
    grid = [c for c, _ in matrix if c.radius_m is not None]
    rotated = [c for c, _ in matrix if c.radius_m is None]
    assert len(grid) == 45 and len(rotated) == 70
    assert all(.14 <= c.radius_m <= .22 and abs(c.azimuth_deg) <= 40 for c in grid)
    assert all(c.yaw_offset_deg == 0 for c in grid)
    assert len({s for c, s in matrix if c.radius_m is None}) == 10


def test_nominal_diagnostic_reproduces_full_rollout() -> None:
    """The placement injection must not perturb normal observations or physics."""
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset, make_env

    env, reference = make_env(CartesianZeroPreset()), make_env(CartesianZeroPreset())
    raw = env.unwrapped
    raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z, PoseCase("normal"))
    try:
        actual, _ = env.reset(seed=82000)
        expected, _ = reference.reset(seed=82000)
        np.testing.assert_array_equal(actual, expected)
        for _ in range(200):
            actual, *_ = env.step(np.zeros(6))
            expected, *_ = reference.step(np.zeros(6))
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(raw.backend.data.qpos, reference.unwrapped.backend.data.qpos)
    finally:
        env.close()
        reference.close()


def test_initial_observation_does_not_reveal_cube_yaw() -> None:
    """Different cube orientations are indistinguishable before contact in this API."""
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset, make_env

    a, b = make_env(CartesianZeroPreset()), make_env(CartesianZeroPreset())
    raw = b.unwrapped
    raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z,
                                  PoseCase("rotated", yaw_offset_deg=30))
    try:
        obs_a, _ = a.reset(seed=82000)
        obs_b, _ = b.reset(seed=82000)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert not np.array_equal(a.unwrapped.backend.data.qpos, raw.backend.data.qpos)
    finally:
        a.close()
        b.close()

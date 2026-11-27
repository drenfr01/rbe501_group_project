"""Placement tracking correction must repair the last timeout without relaxing release guards."""

from dataclasses import replace

import numpy as np

from rbe_501_so101.control.tracked_place import TrackedCarryPlace


def test_last_recorded_timeout_completes_with_unchanged_support_gates() -> None:
    """The 5g/+30deg/89000 cube must be supported before it is released and left behind."""
    from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset
    from scripts.sim2real.analyze.pick_lift_coverage import DiagnosticPlacer, PoseCase

    preset = CartesianZeroPreset()
    preset = replace(preset, base=replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=.005)))
    env = make_timed_env(preset)
    raw = env.unwrapped
    raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z,
                                  PoseCase("last_timeout", yaw_offset_deg=30))
    sequence = TrackedCarryPlace(env)
    try:
        assert sequence.acquire(89000)
        assert sequence.run_transfer(), sequence.failure
        assert sequence.cfg.xy_tolerance_m == .005
        assert sequence.cfg.lower_steps == 150
        assert sequence.support.maximum_descent_m == .003
        assert np.max(np.abs(sequence.tracking_bias)) <= sequence.tracking.correction_limit_m
        assert sequence.frames[-1].supported and not sequence.frames[-1].grasped
        assert next(f for f in sequence.frames if f.phase == "release").supported
        assert not any(f.table_contact or f.warning for f in sequence.frames)
    finally:
        env.close()


def test_carry_matches_previous_candidate_exactly() -> None:
    """Restrict the change to lowering; acquisition and high transport must remain identical."""
    from rbe_501_so101.control.carry_place import TransferPhase
    from rbe_501_so101.control.supported_place import SupportedCarryPlace
    from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset

    a, b = make_timed_env(CartesianZeroPreset()), make_timed_env(CartesianZeroPreset())
    old, new = SupportedCarryPlace(a), TrackedCarryPlace(b)
    try:
        for sequence in (old, new):
            assert sequence.acquire(89000)
            goal = sequence.target.copy()
            goal[2] = sequence.physical.grasp_state().obj.p_s[2]
            assert sequence.move_cube(goal, TransferPhase.CARRY, sequence.cfg.carry_steps)
        np.testing.assert_array_equal(a.unwrapped.backend.data.qpos, b.unwrapped.backend.data.qpos)
        np.testing.assert_array_equal(a.unwrapped.backend.data.qvel, b.unwrapped.backend.data.qvel)
        np.testing.assert_array_equal(new.tracking_bias, np.zeros(3))
    finally:
        a.close()
        b.close()


def test_tracking_does_not_integrate_after_support_is_ready() -> None:
    """Stable support should stop additional correction instead of pressing farther downward."""
    from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset

    env = make_timed_env(CartesianZeroPreset())
    sequence = TrackedCarryPlace(env)
    try:
        sequence.reference = np.zeros(3)
        sequence.tracking_bias[:] = [.001, -.001, -.002]
        sequence.support_ready = lambda: True
        before = sequence.tracking_bias.copy()
        sequence.update_tracking_bias(np.array([0., 0., -.01]))
        np.testing.assert_array_equal(sequence.tracking_bias, before)
    finally:
        env.close()

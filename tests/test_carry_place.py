"""Carry/place must require table support before intentional grasp release."""

import numpy as np

from rbe_501_so101.control.carry_place import TransferConfig, destination, release_ready


def test_destination_is_distinct_and_preserves_reachable_radius() -> None:
    """Move toward a nearby azimuth without extending the arm's radial workspace."""
    axis = np.array([-.16, 0.])
    source = np.array([.02, .04, .4875])
    target = destination(source, axis, .4875, TransferConfig())
    np.testing.assert_allclose(np.linalg.norm(target[:2] - axis), np.linalg.norm(source[:2] - axis))
    assert np.linalg.norm(target[:2] - source[:2]) > .025
    assert target[2] == .4875


def test_release_requires_support_position_and_low_speed() -> None:
    """A cube near the target but hanging in the fingers cannot count as placed."""
    cfg = TransferConfig()
    assert not release_ready(False, .001, .001, .001, cfg)
    assert not release_ready(True, .02, .001, .001, cfg)
    assert not release_ready(True, .001, .02, .001, cfg)
    assert not release_ready(True, .001, .001, .1, cfg)
    assert release_ready(True, .001, .001, .001, cfg)


def test_recorded_cube_is_carried_released_and_left_on_table() -> None:
    """Require physical placement after retreat, not merely moving while still grasped."""
    from dataclasses import replace
    from rbe_501_so101.control.carry_place import CarryPlace
    from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset

    preset = CartesianZeroPreset()
    preset = replace(preset, base=replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=.01)))
    env = make_timed_env(preset)
    sequence = CarryPlace(env)
    try:
        assert sequence.acquire(87000)
        assert sequence.run_transfer(), sequence.failure
        assert sequence.frames[-1].supported
        assert not sequence.frames[-1].grasped
        assert not any(f.table_contact or f.warning for f in sequence.frames)
    finally:
        env.close()


def test_unsupported_release_is_rejected_without_opening() -> None:
    """An acquired airborne cube cannot trigger the intentional-release phase."""
    from rbe_501_so101.control.carry_place import CarryPlace
    from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset

    env = make_timed_env(CartesianZeroPreset())
    sequence = CarryPlace(env)
    try:
        assert sequence.acquire(87000)
        before = env.unwrapped.backend.grasp_state().robot.q_cmd.copy()
        assert not sequence.release()
        assert sequence.failure == "release_without_support"
        np.testing.assert_array_equal(before, env.unwrapped.backend.grasp_state().robot.q_cmd)
    finally:
        env.close()

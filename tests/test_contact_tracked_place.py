"""Once supported, correct lateral placement without continuing downward integration."""

from dataclasses import replace

import numpy as np
import pytest

from rbe_501_so101.control.contact_tracked_place import ContactTrackedCarryPlace


@pytest.mark.parametrize("seed,mass,yaw", [(88000, .015, -30), (89000, .005, 30)])
def test_contact_aware_tracking_repairs_both_recorded_failures(seed, mass, yaw) -> None:
    """Retain the old last timeout repair and avoid the new supported-clearance regression."""
    from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset
    from scripts.sim2real.analyze.pick_lift_coverage import DiagnosticPlacer, PoseCase

    preset = CartesianZeroPreset()
    preset = replace(preset, base=replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass)))
    env = make_timed_env(preset)
    raw = env.unwrapped
    raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z,
                                  PoseCase("recorded", yaw_offset_deg=yaw))
    sequence = ContactTrackedCarryPlace(env)
    try:
        assert sequence.acquire(seed)
        assert sequence.run_transfer(), sequence.failure
        assert sequence.frames[-1].supported and not sequence.frames[-1].grasped
        assert next(f for f in sequence.frames if f.phase == "release").supported
        assert not any(f.table_contact or f.warning for f in sequence.frames)
        assert np.max(np.abs(sequence.tracking_bias)) <= sequence.tracking.correction_limit_m
    finally:
        env.close()


def test_supported_cube_keeps_vertical_bias_fixed_while_xy_is_corrected() -> None:
    """Table support must stop downward integration even before lateral readiness is met."""
    from types import SimpleNamespace
    from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset

    env = make_timed_env(CartesianZeroPreset())
    sequence = ContactTrackedCarryPlace(env)
    try:
        goal = np.array([.1, .2, .48])
        sequence.tracking_bias[:] = [.001, -.001, -.004]
        sequence.reference = goal + sequence.tracking_bias
        sequence.contacts = lambda: (True, False)
        sequence.support_ready = lambda: False
        sequence.backend.grasp_state = lambda: SimpleNamespace(p_pinch=goal + [.002, -.002, .01])
        before = sequence.tracking_bias.copy()
        sequence.update_tracking_bias(goal)
        assert sequence.tracking_bias[2] == before[2]
        assert not np.array_equal(sequence.tracking_bias[:2], before[:2])
    finally:
        env.close()

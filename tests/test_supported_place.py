"""A bounded support probe must repair hovering without weakening release authorization."""

from dataclasses import replace

from rbe_501_so101.control.supported_place import SupportedCarryPlace


def test_recorded_light_cube_hover_reaches_support_before_release() -> None:
    """A previously held 5 g cube must actually touch down rather than release in air."""
    from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset
    from scripts.sim2real.analyze.pick_lift_coverage import DiagnosticPlacer, PoseCase

    preset = CartesianZeroPreset()
    preset = replace(preset, base=replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=.005)))
    env = make_timed_env(preset)
    raw = env.unwrapped
    raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z, PoseCase("normal"))
    sequence = SupportedCarryPlace(env)
    try:
        assert sequence.acquire(88002)
        assert sequence.run_transfer(), sequence.failure
        release = [f for f in sequence.frames if f.phase == "release"]
        assert release and release[0].supported
        assert sequence.frames[-1].supported and not sequence.frames[-1].grasped
        assert not any(f.table_contact or f.warning for f in sequence.frames)
    finally:
        env.close()

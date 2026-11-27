"""Bounded settling must preserve measured readiness, closing dwell and contact gates."""

import numpy as np

from rbe_501_so101.control.scripted_pick_lift import Phase
from rbe_501_so101.control.timed_oracle_acquisition import TimingConfig, make_timed_env, settling_phase


def test_settling_has_a_deadline_and_preserves_real_time_dwell() -> None:
    """Wait without advancing into lift prematurely, but never wait forever."""
    cfg = TimingConfig()
    assert settling_phase(99, Phase.CLOSE, None, 25, cfg) == Phase.CLOSE
    assert settling_phase(100, Phase.LIFT, None, 25, cfg) == Phase.CLOSE
    assert settling_phase(119, Phase.LIFT, None, 25, cfg) == Phase.CLOSE
    assert settling_phase(120, Phase.LIFT, None, 25, cfg) == Phase.LIFT
    assert settling_phase(142, Phase.LIFT, 118, 25, cfg) == Phase.CLOSE
    assert settling_phase(143, Phase.LIFT, 118, 25, cfg) == Phase.LIFT


def test_recorded_grid_timeout_is_repaired_without_relaxing_gates() -> None:
    """The arm must keep settling after step 100 and lift within the original horizon."""
    from dataclasses import replace
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset
    from scripts.sim2real.analyze.pick_lift_coverage import DiagnosticPlacer, PoseCase

    preset = CartesianZeroPreset()
    preset = replace(preset, base=replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=.01)))
    env = make_timed_env(preset)
    raw = env.unwrapped
    raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z, PoseCase("regression", .20, -40))
    try:
        env.reset(seed=83000)
        released_at = None
        for step in range(200):
            _, _, _, truncated, info = env.step(np.zeros(6))
            if info["lift_released"] and released_at is None:
                released_at = step
                assert raw.grasp_state.grasped
        assert truncated and info["is_success"]
        assert 100 < released_at <= 150
        assert env._controller.close_start_step > 100
        assert env._controller.acquisition.stable_alignment_steps == 3
        assert env._controller.descent.close_dwell_steps == 25
        assert info["robot_table_contact_steps"] == 0
    finally:
        env.close()


def test_alignment_failure_aborts_open_at_deadline() -> None:
    """Extra settling time must not bypass failed measured alignment or freeze real time."""
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset

    env = make_timed_env(CartesianZeroPreset())
    try:
        env.reset(seed=82000)
        controller = env._controller
        controller._alignment = lambda obs: False
        for step in range(200):
            env.step(np.zeros(6))
            assert controller._step == step + 1
            assert not controller.close_started
            assert not env.env.env.gate.released
            if step < controller.timing.alignment_deadline_step:
                assert not controller.aborted
        assert controller.aborted and controller.abort_reason == "alignment_timeout"
        np.testing.assert_allclose(env.unwrapped.grasp_state.robot.q_cmd[-1], np.deg2rad(45))
    finally:
        env.close()


def test_missing_contacts_cannot_release_lift() -> None:
    """A closing command cannot substitute for sensed stable two-pad contact."""
    from dataclasses import replace
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset

    env = make_timed_env(CartesianZeroPreset())
    backend = env.unwrapped.backend
    original = backend.grasp_state
    backend.grasp_state = lambda: replace(original(), grasped=False)
    try:
        env.reset(seed=82000)
        for _ in range(200):
            env.step(np.zeros(6))
            assert not env.env.env.gate.released
        assert env._controller.close_started
        assert env._controller.abort_reason == "grasp_deadline"
    finally:
        env.close()

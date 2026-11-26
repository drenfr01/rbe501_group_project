"""Holding probes need exact force timing, restoration and physically defined metrics."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.sim2real.analyze.acquisition_holding_checks import Pulse, force_at, temporary_force, displacement_rms


def test_force_pulse_timing_and_restore_after_failure():
    """Apply force for exactly three control periods and preserve prior applied wrench."""
    pulse = Pulse("x", (.1, 0., 0.))
    np.testing.assert_array_equal(force_at(149, pulse), np.zeros(3))
    for step in (150, 151, 152):
        np.testing.assert_array_equal(force_at(step, pulse), [.1, 0, 0])
    np.testing.assert_array_equal(force_at(153, pulse), np.zeros(3))
    data = SimpleNamespace(xfrc_applied=np.ones((2, 6)))
    with pytest.raises(RuntimeError):
        with temporary_force(data, 1, np.array([.1, 0, 0])):
            np.testing.assert_array_equal(data.xfrc_applied[1], [1.1, 1, 1, 1, 1, 1])
            raise RuntimeError("probe interrupted")
    np.testing.assert_array_equal(data.xfrc_applied, np.ones((2, 6)))


def test_displacement_rms_uses_fixed_pre_pulse_reference():
    """Do not confuse motion relative to the held pose with reach-target IK error."""
    assert displacement_rms(np.array([[.003, .004, 0], [.003, .004, 0]]), np.zeros(3)) == pytest.approx(.005)


def test_world_force_produces_expected_free_body_impulse():
    """Check the force units and target-body application using an isolated free cube."""
    import mujoco

    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.001" gravity="0 0 0"/>
        <worldbody><body name="cube" pos="0 0 1"><freejoint/>
        <geom type="box" size=".01 .01 .01" mass=".01"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    with temporary_force(data, model.body("cube").id, np.array([.1, 0, 0])):
        for _ in range(10):
            mujoco.mj_step(model, data)
    assert data.qvel[0] == pytest.approx(.1)
    np.testing.assert_array_equal(data.xfrc_applied, np.zeros_like(data.xfrc_applied))


def test_paired_metric_subtracts_each_controllers_unforced_drift():
    """Compare common eligible placements and do not mistake baseline drift for recovery."""
    from scripts.sim2real.analyze.acquisition_holding_checks import paired_results

    rows = [{"controller": c, "mass_kg": .01, "seed": 1, "pulse": p, "eligible": True,
             "success": True, "metrics": {"cube_rms_mm": value}}
            for c, p, value in (("scripted", "none", 2), ("limited_sac", "none", 3),
                                ("scripted", ".1N_+x", 5), ("limited_sac", ".1N_+x", 5))]
    pair = paired_results(rows)[0]
    assert pair["raw_cube_rms_difference_mm"] == 0
    assert pair["adjusted_cube_rms_difference_mm"] == -1
    rows[-1]["eligible"] = False
    assert paired_results(rows) == []


def test_zero_force_rollout_matches_existing_baseline():
    """Additional measurement calls and force restoration must not change physics."""
    from scripts.sim2real.analyze.acquisition_holding_checks import holding_episode, HoldingProtocol, fixed_env
    from scripts.sim2real.analyze.calibrated_lift_pilot import ZeroResidual
    from scripts.sim2real.analyze.pick_lift_calibration import evaluate_episode
    from rbe_501_so101.rl.limited_lift_training import LimitedLiftPreset

    env, baseline = fixed_env(LimitedLiftPreset(), .01), fixed_env(LimitedLiftPreset(), .01)
    try:
        actual = holding_episode(env, ZeroResidual(), 68000, Pulse("none", (0., 0., 0.)), HoldingProtocol())
        expected = evaluate_episode(baseline, ZeroResidual(), 68000)
        np.testing.assert_array_equal(env.unwrapped.backend.data.qpos, baseline.unwrapped.backend.data.qpos)
        assert actual["success"] == expected.final_success
        assert actual["eligible"]
    finally:
        env.close()
        baseline.close()

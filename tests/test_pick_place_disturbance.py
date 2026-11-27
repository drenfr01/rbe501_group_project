"""Carry-force evaluation must preserve pairing, physical timing and valid recovery evidence."""

import numpy as np

from scripts.sim2real.analyze.pick_place_disturbance import (
    CarryOutcome, CarryProtocol, CarryPulse, assess, comparisons, eligible, recovery_periods, pulse_force, pulses,
)


def test_pulse_has_exact_three_period_duration() -> None:
    """The same absolute world force schedule must be used for both controllers."""
    cfg, pulse = CarryProtocol(), CarryPulse("push", (.1, 0., 0.))
    for step in range(cfg.pulse_step - 1, cfg.pulse_step + cfg.pulse_periods + 1):
        expected = [.1, 0, 0] if cfg.pulse_step <= step < cfg.pulse_step + 3 else [0, 0, 0]
        np.testing.assert_array_equal(pulse_force(step, pulse, cfg), expected)


def test_force_requires_established_carrying_grasp() -> None:
    """Do not turn an acquisition failure or supported cube into a disturbance trial."""
    cfg = CarryProtocol()
    assert eligible("carry", [True] * cfg.stable_periods, cfg)
    assert not eligible("lower", [True] * cfg.stable_periods, cfg)
    assert not eligible("carry", [True, False, True, True, True], cfg)


def test_recovery_requires_stable_grasped_return_to_nominal_path() -> None:
    """A released/dropped cube cannot be counted as recovered merely because it stops moving."""
    cfg = CarryProtocol()
    assert recovery_periods([0.] * 5, [True] * 5, cfg) == 5
    assert recovery_periods([0.] * 5, [False] * 5, cfg) is None
    assert recovery_periods([.01, 0., 0., 0., 0.], [True] * 5, cfg) is None


def test_failure_during_partial_pulse_remains_in_primary_comparison() -> None:
    """Do not censor a drop because it prevents delivery of all three scheduled periods."""
    cfg = CarryProtocol(placements=1, masses_kg=(.005,), forces_n=(.1,), bootstrap_samples=10)
    rows = [CarryOutcome(c, cfg.first_seed, .005, p.name, True, True, 2., 0, False,
                        False, 3, .099, ("carry",) * 3, "", 450, 0, (), (), ())
            for c in ("scripted", "sac") for p in pulses(cfg)]
    from dataclasses import replace
    position = next(i for i, r in enumerate(rows) if r.controller == "sac" and r.pulse != "none")
    rows[position] = replace(rows[position], success=False, force_periods=1, force_duration_s=.033,
                             pulse_phases=("carry",), transport_grasp_loss=True,
                             failure_reason="grasp_lost_during_transport")
    pairs = comparisons(rows, cfg)
    result = assess(rows, pairs, cfg)
    assert result["common_eligible_disturbed_pairs"] == 4
    assert result["additional_rl_failures"] == 1
    assert not result["useful_benefit_established"]


def test_no_force_instrumentation_preserves_full_task_physics() -> None:
    """Measuring carry windows and restoring zero force must not change the baseline trajectory."""
    from rbe_501_so101.envs.pick_place_env import SO101PickPlaceEnv
    from scripts.sim2real.analyze.pick_place_disturbance import episode
    from scripts.sim2real.analyze.pick_place_pilot import ZeroPolicy, episode as baseline_episode

    env, reference = SO101PickPlaceEnv(), SO101PickPlaceEnv()
    try:
        result = episode(env, ZeroPolicy(), "scripted", 94000, CarryPulse("none", (0., 0., 0.)), CarryProtocol())
        baseline = baseline_episode(reference, ZeroPolicy(), 94000, 0.)
        np.testing.assert_array_equal(env.sequence.backend.data.qpos, reference.sequence.backend.data.qpos)
        np.testing.assert_array_equal(env.sequence.backend.data.qvel, reference.sequence.backend.data.qvel)
        assert result.success == baseline["success"]
        assert result.eligible and result.force_periods == 3
        assert not np.any(env.sequence.backend.data.xfrc_applied)
    finally:
        env.close()
        reference.close()

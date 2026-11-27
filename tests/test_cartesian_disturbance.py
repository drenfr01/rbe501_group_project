"""Paired robustness evaluation must preserve failures and count acquisition separately."""

from dataclasses import replace

import numpy as np

from scripts.sim2real.analyze.cartesian_disturbance import (
    DisturbanceOutcome, RobustnessProtocol, assess, episode, pulses, recovery_steps,
)


def outcome(controller: str, seed: int, pulse: str = "0.4N_+x") -> DisturbanceOutcome:
    """Build a common eligible, retained hold for assessment tests."""
    return DisturbanceOutcome(controller, .005, seed, pulse, True, True, False,
                              False, False, 0, False, 10, .2, .01, .5, .1)


def test_transient_failure_recovery_requires_consecutive_success() -> None:
    """A single successful sample must not be mistaken for settled recovery."""
    assert recovery_steps([False, True, False, True, True, True], 3) == 6
    assert recovery_steps([False, True, True], 3) is None


def test_paired_failure_blocks_benefit_even_with_lower_motion() -> None:
    """Precision in retained holds cannot hide an extra RL drop."""
    rows = [outcome(c, s, p) for c in ("scripted", "sac")
            for s in range(3) for p in ("none", "0.4N_+x")]
    rows[-1] = replace(rows[-1], success=False, lost_success=True, lost_grasp=True,
                       cube_rms_mm=.01)
    result = assess(rows, RobustnessProtocol())
    assert result["additional_rl_failures"] == 1
    assert not result["retention_not_worse"]
    assert not result["useful_benefit_established"]


def test_ineligible_acquisition_is_not_a_disturbance_win() -> None:
    """When acquisition differs, keep outcomes but exclude the force-retention pair."""
    rows = [outcome("scripted", 81000),
            replace(outcome("sac", 81000), eligible=False, success=False)]
    result = assess(rows, RobustnessProtocol())
    assert result["common_eligible_disturbed_pairs"] == 0
    assert result["overall_final_successes"] == {"scripted": 1, "sac": 0}
    assert not result["useful_benefit_established"]


def test_zero_force_probe_preserves_v3_physics_exactly() -> None:
    """Evaluation instrumentation must reproduce the original unforced trajectory."""
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset, make_env
    from scripts.sim2real.analyze.calibrated_lift_pilot import ZeroResidual

    env, reference = make_env(CartesianZeroPreset()), make_env(CartesianZeroPreset())
    protocol = RobustnessProtocol()
    try:
        result = episode(env, ZeroResidual(), "scripted", .01, 81000, pulses(protocol)[0], protocol)
        reference.reset(seed=81000)
        for _ in range(reference.unwrapped.config.task.max_episode_steps):
            reference.step(np.zeros(6))
        actual, expected = env.unwrapped.backend.data, reference.unwrapped.backend.data
        np.testing.assert_array_equal(actual.qpos, expected.qpos)
        np.testing.assert_array_equal(actual.qvel, expected.qvel)
        assert result.eligible and result.success
        assert not np.any(actual.xfrc_applied)
    finally:
        env.close()
        reference.close()

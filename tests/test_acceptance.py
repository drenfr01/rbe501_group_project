"""Acceptance criteria for a retrained policy, written down before training (plan step 15)."""

import pytest

from rbe_501_so101.config import PRESETS, AcceptanceConfig
from rbe_501_so101.rl.acceptance import (
    DEPLOYMENT,
    STICTION_SCALE,
    acceptance_conditions,
    check_acceptance,
)
from rbe_501_so101.rl.robustness import KP_SCALE, TRAINED, SweepResult

CFG = AcceptanceConfig(kp_scales=(0.7, 1.4), stiction_scales=(1.5,))
TRAINING = PRESETS["residual_ik_sim2real"].env


def _result(controller: str, parameter: str, value: float, mean: float, saturation: float | None = None,
            spread: float = 0.1, command_step: float = 0.02) -> SweepResult:
    """One controller's result under one condition."""
    seeds = {"0": mean - spread / 2, "1": mean + spread / 2} if controller == TRAINED else {"all": mean}
    return SweepResult(controller, parameter, value, mean, 2 * mean, seeds, saturation, command_step)


def _results(rl: dict[tuple[str, float], float], integral: float = 1.0, ik: float = 10.0,
             saturation: float = 0.05, command_step: float = 0.02) -> list[SweepResult]:
    """Trained policy at the given means, with fixed baselines under every condition."""
    out = []
    for (parameter, value), mean in rl.items():
        out.append(_result(TRAINED, parameter, value, mean, saturation, command_step=command_step))
        out += [_result("integral", parameter, value, integral), _result("ik_hold", parameter, value, ik),
                _result("gravity_comp", parameter, value, 5.0)]
    return out


GOOD = {(DEPLOYMENT, 1.0): 0.8, (KP_SCALE, 0.7): 0.9, (KP_SCALE, 1.4): 0.9, (STICTION_SCALE, 1.5): 0.95}


def test_conditions_are_deployment_then_each_stress_case():
    """Deployment first, then every kp and stiction extreme, all with randomization fixed."""
    conditions = acceptance_conditions(TRAINING, CFG, speed_scale=0.5)
    assert [(c.parameter, c.value) for c in conditions] == list(GOOD)
    assert all(c.speed_scale == 0.5 for c in conditions)
    kp_low = conditions[1].options.randomization
    assert kp_low.enabled and kp_low.kp_scale_range == (0.7, 0.7) and kp_low.payload_mass_range_kg == (0.0, 0.0)
    assert conditions[3].options.servo.stiction_scale_range == (1.5, 1.5)
    assert not conditions[3].options.randomization.enabled


def test_a_policy_meeting_every_criterion_passes():
    """Every check passes and each reports a reason."""
    verdict = check_acceptance(_results(GOOD), CFG)
    assert verdict.passed
    assert all(r.startswith("PASS") for r in verdict.reasons)


@pytest.mark.parametrize(
    ("change", "failing"),
    [
        ({(DEPLOYMENT, 1.0): 2.5}, "deployment"),
        ({(KP_SCALE, 0.7): 6.0}, "worst case"),
        ({(STICTION_SCALE, 1.5): 1.2}, "integral"),
    ],
)
def test_each_criterion_can_fail_on_its_own(change, failing):
    """Breaking one criterion fails the verdict, with that criterion named."""
    verdict = check_acceptance(_results(GOOD | change), CFG)
    assert not verdict.passed
    assert any(failing in r for r in verdict.reasons if r.startswith("FAIL"))


def test_saturation_and_ik_hold_margin_are_checked():
    """Too much saturation, or not beating IK hold by more than the seed spread, fails."""
    assert not check_acceptance(_results(GOOD, saturation=0.3), CFG).passed
    assert not check_acceptance(_results(GOOD, ik=0.85), CFG).passed


def test_missing_trained_results_are_refused():
    """Baseline-only results cannot be judged."""
    baselines = [r for r in _results(GOOD) if r.controller != TRAINED]
    with pytest.raises(ValueError, match="trained"):
        check_acceptance(baselines, CFG)


def test_dithering_fails_even_when_accurate():
    """A policy that moves its command every step while holding is rejected (wear, unvalidated regime)."""
    verdict = check_acceptance(_results(GOOD, command_step=1.9), CFG)
    assert not verdict.passed
    assert any("dither" in r for r in verdict.reasons if r.startswith("FAIL"))

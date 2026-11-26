"""Plan steps 14-15: the deployment condition, and the bar a retrained policy must clear before hardware."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from rbe_501_so101.config import AcceptanceConfig, DynamicsRandomizationConfig, EnvOptions
from rbe_501_so101.rl.robustness import KP_SCALE, TRAINED, SweepCondition, SweepResult, fixed_kp_options

DEPLOYMENT = "deployment"
STICTION_SCALE = "stiction_scale"
INTEGRAL, IK_HOLD = "integral", "ik_hold"


@dataclass(frozen=True)
class AcceptanceVerdict:
    """Whether every criterion holds, with one PASS/FAIL line per criterion."""

    passed: bool
    reasons: tuple[str, ...]


def deployment_options(training: EnvOptions) -> EnvOptions:
    """The training options with every random draw fixed at the measured arm.

    Calibrated stiffness, nominal stiction band, the longest training delay
    every step, encoder rounding, LeRobot's clip, and the real joint window.
    """
    servo = replace(training.servo, stiction_scale_range=(1.0, 1.0))
    perturbation = replace(training.perturbation, randomize_delay=False)
    return replace(
        training, randomization=DynamicsRandomizationConfig(enabled=False), servo=servo, perturbation=perturbation
    )


def stiction_options(scale: float, base: EnvOptions) -> EnvOptions:
    """Base options with the stiction band scaled by exactly `scale`."""
    return replace(base, servo=replace(base.servo, stiction_scale_range=(scale, scale)))


def acceptance_conditions(training: EnvOptions, cfg: AcceptanceConfig, speed_scale: float = 1.0) -> list[SweepCondition]:
    """Deployment, then each kp extreme, then each stiction extreme (all at `speed_scale`)."""
    base = deployment_options(training)
    conditions = [SweepCondition(DEPLOYMENT, 1.0, base, speed_scale)]
    conditions += [SweepCondition(KP_SCALE, s, fixed_kp_options(s, base), speed_scale) for s in cfg.kp_scales]
    conditions += [SweepCondition(STICTION_SCALE, s, stiction_options(s, base), speed_scale) for s in cfg.stiction_scales]
    return conditions


def check_acceptance(results: Sequence[SweepResult], cfg: AcceptanceConfig) -> AcceptanceVerdict:
    """Apply every criterion of `AcceptanceConfig` to the results of `acceptance_conditions`."""
    trained = [r for r in results if r.controller == TRAINED]
    if not trained:
        raise ValueError("acceptance needs trained-policy results")
    deploy = next(r for r in trained if r.parameter == DEPLOYMENT)
    checks = (
        _deployment_check(deploy, cfg),
        _ik_hold_check(deploy, _baseline(results, IK_HOLD, deploy)),
        _saturation_check(deploy, cfg),
        _dither_check(deploy, cfg),
        _worst_case_check([r for r in trained if r.parameter != DEPLOYMENT], cfg),
        _integral_check(trained, results, cfg),
    )
    reasons = tuple(f"{'PASS' if ok else 'FAIL'}: {text}" for ok, text in checks)
    return AcceptanceVerdict(all(ok for ok, _ in checks), reasons)


def _baseline(results: Sequence[SweepResult], name: str, like: SweepResult) -> SweepResult:
    """A baseline's result under the same condition as `like`."""
    return next(r for r in results if r.controller == name and (r.parameter, r.value) == (like.parameter, like.value))


def _deployment_check(deploy: SweepResult, cfg: AcceptanceConfig) -> tuple[bool, str]:
    """Holding error in the deployment condition."""
    return deploy.mean_mm <= cfg.deployment_max_mean_mm, (
        f"deployment mean error {deploy.mean_mm:.2f} mm vs limit {cfg.deployment_max_mean_mm:.2f} mm"
    )


def _ik_hold_check(deploy: SweepResult, ik: SweepResult) -> tuple[bool, str]:
    """Margin over IK hold, against the spread between seeds."""
    spread, margin = float(np.ptp(list(deploy.per_seed_mean_mm.values()))), ik.mean_mm - deploy.mean_mm
    return margin > spread, f"beats IK hold by {margin:.2f} mm vs seed spread {spread:.2f} mm"


def _saturation_check(deploy: SweepResult, cfg: AcceptanceConfig) -> tuple[bool, str]:
    """Fraction of holding steps at the action limit."""
    saturation = deploy.saturation_fraction or 0.0
    return saturation < cfg.max_saturation_fraction, (
        f"saturation {saturation:.1%} vs limit {cfg.max_saturation_fraction:.0%}"
    )


def _dither_check(deploy: SweepResult, cfg: AcceptanceConfig) -> tuple[bool, str]:
    """Mean command change per holding step (a still hold, not dithering against friction)."""
    if deploy.command_step_deg is None:
        return False, "dither: command step not measured"
    return deploy.command_step_deg <= cfg.max_command_step_deg, (
        f"dither: command moves {deploy.command_step_deg:.2f} deg/step while holding vs limit {cfg.max_command_step_deg:.2f}"
    )


def _worst_case_check(stress: Sequence[SweepResult], cfg: AcceptanceConfig) -> tuple[bool, str]:
    """Largest mean error over the stress conditions."""
    worst = max(stress, key=lambda r: r.mean_mm)
    return worst.mean_mm <= cfg.worst_case_max_mean_mm, (
        f"worst case {worst.parameter} {worst.value:g}: {worst.mean_mm:.2f} mm vs limit {cfg.worst_case_max_mean_mm:.2f} mm"
    )


def _integral_check(
    trained: Sequence[SweepResult], results: Sequence[SweepResult], cfg: AcceptanceConfig
) -> tuple[bool, str]:
    """Policy minus integral control, in the condition where the policy does worst relative to it."""
    gaps = [(r.mean_mm - _baseline(results, INTEGRAL, r).mean_mm, r) for r in trained]
    gap, r = max(gaps, key=lambda g: g[0])
    return gap <= cfg.integral_margin_mm, (
        f"vs integral control, worst at {r.parameter} {r.value:g}: {gap:+.2f} mm (allowed {cfg.integral_margin_mm:+.2f} mm)"
    )

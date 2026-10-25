"""Phase 2: what the sim policy depends on (stiffness, sensing, timing, speed) and the transfer rule.

Every condition evaluates the trained policy (in its preset's residual env)
and the three classical baselines (in a delta-mode env with the same task
timing and dynamics) on the same evaluation targets.
"""

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from rbe_501_so101.config import (
    DynamicsRandomizationConfig,
    EnvOptions,
    ExperimentPreset,
    PerturbationConfig,
    ReachTaskConfig,
    RobustnessConfig,
    SimConfig,
    TrainConfig,
    resolve_project_path,
)
from rbe_501_so101.envs.calibration import ActuatorCalibration
from rbe_501_so101.envs.types import ObservationLayout
from rbe_501_so101.rl.checkpoints import CheckpointRepository
from rbe_501_so101.rl.evaluation import M_TO_MM, EpisodeStats, evaluate, load_trained_policy, make_eval_env, summarize
from rbe_501_so101.rl.policies import BASELINE_NAMES, make_baseline

TRAINED = "trained"
KP_SCALE = "kp_scale"
NOMINAL_SCALE = 1.0
FEEDBACK_SPLIT = 0.5
MIN_LOG_SEPARATION = 0.1
CorrectionKind = Literal["feedback", "feedforward", "inconclusive"]


@dataclass(frozen=True)
class SweepCondition:
    """One evaluated setting: a named parameter value, env options, and command-rate scale."""

    parameter: str
    value: float
    options: EnvOptions
    speed_scale: float = 1.0


@dataclass(frozen=True)
class SweepResult:
    """Holding-window error of one controller under one condition (mm)."""

    controller: str
    parameter: str
    value: float
    mean_mm: float
    p95_mm: float
    per_seed_mean_mm: dict[str, float]
    saturation_fraction: float | None


@dataclass(frozen=True)
class TransferDecision:
    """Verdict of the transfer-or-retrain rule and one reason per check."""

    transfer: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CorrectionVerdict:
    """How the policy's error grows with stiffness mismatch, relative to the two baselines.

    `position` is 0 when the policy degrades like integral control and 1 when
    it degrades like gravity compensation (log-growth scale).
    """

    kind: CorrectionKind
    trained_growth: float
    integral_growth: float
    gravity_comp_growth: float
    position: float | None


def fixed_kp_options(scale: float, base: EnvOptions = EnvOptions()) -> EnvOptions:
    """Base options with every actuator stiffness scaled by exactly `scale` (no other randomization)."""
    randomization = DynamicsRandomizationConfig(
        enabled=True, kp_scale_range=(scale, scale), damping_scale_range=(1.0, 1.0), payload_mass_range_kg=(0.0, 0.0)
    )
    return replace(base, randomization=randomization)


def sensing_options(delay_steps: int, cfg: RobustnessConfig, base: EnvOptions = EnvOptions()) -> EnvOptions:
    """Base options plus encoder rounding and a fixed whole-step delay."""
    return replace(base, perturbation=PerturbationConfig(delay_steps=delay_steps, quantization_rad=cfg.quantization_rad))


def saturation_fraction(actions: np.ndarray, threshold: float) -> float:
    """Fraction of steps where any action component reaches `threshold` in magnitude."""
    return float(np.mean(np.any(np.abs(np.asarray(actions)) >= threshold, axis=1)))


def run_stiffness_sweep(
    repo: CheckpointRepository,
    seeds: tuple[int, ...],
    cfg: RobustnessConfig,
    base: EnvOptions = EnvOptions(),
    train_cfg: TrainConfig = TrainConfig(),
) -> list[SweepResult]:
    """Every controller at every `kp` scale in the grid."""
    conditions = [SweepCondition(KP_SCALE, s, fixed_kp_options(s, base)) for s in cfg.kp_scales]
    return [r for c in conditions for r in evaluate_condition(repo, seeds, c, cfg, train_cfg)]


def run_sensing_sweep(
    repo: CheckpointRepository,
    seeds: tuple[int, ...],
    cfg: RobustnessConfig,
    base: EnvOptions = EnvOptions(),
    train_cfg: TrainConfig = TrainConfig(),
) -> list[SweepResult]:
    """Encoder rounding with each configured delay."""
    conditions = [SweepCondition("delay_steps", d, sensing_options(d, cfg, base)) for d in cfg.delay_steps]
    return [r for c in conditions for r in evaluate_condition(repo, seeds, c, cfg, train_cfg)]


def evaluate_condition(
    repo: CheckpointRepository,
    seeds: tuple[int, ...],
    condition: SweepCondition,
    cfg: RobustnessConfig,
    train_cfg: TrainConfig = TrainConfig(),
) -> list[SweepResult]:
    """Trained policy (all seeds) and each baseline (once) under one condition."""
    trained = {str(s): _trained_stats(repo, s, condition, cfg, train_cfg) for s in seeds}
    results = [_to_result(TRAINED, condition, trained, cfg)]
    preset = repo.load_preset(seeds[0])
    for name in BASELINE_NAMES:
        results.append(_to_result(name, condition, {"all": _baseline_stats(name, preset, condition, cfg, train_cfg)}, None))
    return results


def transfer_decision(results: list[SweepResult], cfg: RobustnessConfig) -> TransferDecision:
    """Apply the decision rule to the results of the calibrated, perturbed condition."""
    rl = _find(results, TRAINED)
    ik = _find(results, "ik_hold")
    spread = float(np.ptp(list(rl.per_seed_mean_mm.values())))
    limit_mm = M_TO_MM * cfg.transfer_max_mean_m
    saturation = rl.saturation_fraction or 0.0
    checks = (
        (rl.mean_mm <= limit_mm, f"mean error {rl.mean_mm:.2f} mm vs limit {limit_mm:.2f} mm"),
        (ik.mean_mm - rl.mean_mm > spread, f"beats IK hold by {ik.mean_mm - rl.mean_mm:.2f} mm vs seed spread {spread:.2f} mm"),
        (saturation < cfg.max_saturation_fraction, f"saturation {saturation:.1%} vs limit {cfg.max_saturation_fraction:.0%}"),
    )
    reasons = tuple(f"{'PASS' if ok else 'FAIL'}: {text}" for ok, text in checks)
    return TransferDecision(transfer=all(ok for ok, _ in checks), reasons=reasons)


def classify_correction(results: list[SweepResult]) -> CorrectionVerdict:
    """Feedback if the policy's error growth is closer (log scale) to integral control than to gravity comp."""
    growth = {name: _growth(results, name) for name in (TRAINED, "integral", "gravity_comp")}
    log_int, log_grav = math.log(growth["integral"]), math.log(growth["gravity_comp"])
    if log_grav - log_int < MIN_LOG_SEPARATION:
        return CorrectionVerdict("inconclusive", growth[TRAINED], growth["integral"], growth["gravity_comp"], None)
    position = (math.log(growth[TRAINED]) - log_int) / (log_grav - log_int)
    kind: CorrectionKind = "feedback" if position < FEEDBACK_SPLIT else "feedforward"
    return CorrectionVerdict(kind, growth[TRAINED], growth["integral"], growth["gravity_comp"], position)


def working_range(results: list[SweepResult], cfg: RobustnessConfig) -> tuple[float, float] | None:
    """Contiguous `kp` scales around nominal where the policy meets the transfer error limit."""
    points = sorted((r.value, r.mean_mm) for r in results if r.controller == TRAINED and r.parameter == KP_SCALE)
    ok = [v for v, m in points if m <= M_TO_MM * cfg.transfer_max_mean_m]
    if NOMINAL_SCALE not in ok:
        return None
    values = [v for v, _ in points]
    i = j = values.index(NOMINAL_SCALE)
    while i > 0 and values[i - 1] in ok:
        i -= 1
    while j < len(values) - 1 and values[j + 1] in ok:
        j += 1
    return values[i], values[j]


def baseline_task(task: ReachTaskConfig) -> ReachTaskConfig:
    """Delta-mode version of a task (same rate limit, cap, and episode length) for the baselines."""
    return replace(task, action_mode="delta", fine_error_obs=False)


def measured_kp_scales(calibration: ActuatorCalibration, nominal_kp: np.ndarray, joints: tuple[str, ...]) -> dict[str, float]:
    """Measured / nominal stiffness per calibrated joint (for marking the sweep plot)."""
    measured = calibration.kp_for(joints, nominal_kp)
    return {j: float(m / n) for j, m, n in zip(joints, measured, nominal_kp, strict=True) if m != n}


def _trained_stats(
    repo: CheckpointRepository, seed: int, condition: SweepCondition, cfg: RobustnessConfig, train_cfg: TrainConfig
) -> list[EpisodeStats]:
    """Evaluate one checkpoint seed in its preset env under the condition."""
    preset = slowed_preset(repo.load_preset(seed), condition.speed_scale)
    env = make_eval_env(preset, randomize=False, options=condition.options)
    policy = load_trained_policy(repo, seed, train_cfg.algo)
    try:
        return evaluate(env, policy, cfg.episodes_per_point, train_cfg)
    finally:
        env.close()
        policy.close()


def _baseline_stats(
    name: str, preset: ExperimentPreset, condition: SweepCondition, cfg: RobustnessConfig, train_cfg: TrainConfig
) -> list[EpisodeStats]:
    """Evaluate one classical baseline in a delta-mode env under the condition."""
    task = baseline_task(slowed_preset(preset, condition.speed_scale).task)
    env = make_eval_env(replace(preset, task=task), randomize=False, options=condition.options)
    calibration = _calibration(condition.options)
    policy = make_baseline(name, SimConfig(), task, ObservationLayout(n_arm=env.layout.n_arm), calibration)
    try:
        return evaluate(env, policy, cfg.episodes_per_point, train_cfg)
    finally:
        env.close()


def _to_result(
    controller: str, condition: SweepCondition, per_seed: dict[str, list[EpisodeStats]], cfg: RobustnessConfig | None
) -> SweepResult:
    """Summarize episodes; saturation is only reported for the residual policy."""
    report = summarize(per_seed)
    saturation = None
    if cfg is not None:
        actions = np.concatenate([ep.holding_actions for eps in per_seed.values() for ep in eps])
        saturation = saturation_fraction(actions, cfg.saturation_threshold)
    return SweepResult(
        controller, condition.parameter, condition.value, report.mean_mm, report.p95_mm, report.per_seed_mean_mm, saturation
    )


def slowed_preset(preset: ExperimentPreset, speed_scale: float) -> ExperimentPreset:
    """Preset with the command rate limit scaled (hardware runs slower at first)."""
    if speed_scale == 1.0:
        return preset
    return replace(preset, task=replace(preset.task, delta_q_max_rad=preset.task.delta_q_max_rad * speed_scale))


def _calibration(options: EnvOptions) -> ActuatorCalibration | None:
    """Load the calibration named by the options, if any."""
    if options.calibration_path is None:
        return None
    return ActuatorCalibration.load(resolve_project_path(options.calibration_path))


def _find(results: list[SweepResult], controller: str) -> SweepResult:
    """The single result for a controller."""
    return next(r for r in results if r.controller == controller)


def _growth(results: list[SweepResult], controller: str) -> float:
    """Worst error over the sweep divided by the error at nominal stiffness."""
    points = {r.value: r.mean_mm for r in results if r.controller == controller and r.parameter == KP_SCALE}
    return max(points.values()) / points[NOMINAL_SCALE]

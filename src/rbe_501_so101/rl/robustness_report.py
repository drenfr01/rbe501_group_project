"""Phase 2 experiment record: the typed report, its README, and its plots."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rbe_501_so101.hardware.sag_analysis import JointSagComparison
from rbe_501_so101.rl.plots import plot_kp_sweep, plot_saturation
from rbe_501_so101.rl.robustness import TRAINED, CorrectionVerdict, SweepResult, TransferDecision

CONTROLLER_ORDER = (TRAINED, "ik_hold", "gravity_comp", "integral")
KP_SWEEP_PNG = "kp_sweep.png"
SATURATION_PNG = "saturation.png"


@dataclass(frozen=True)
class RobustnessReport:
    """Every Phase 2 result for one checkpoint, plus the verdicts drawn from them."""

    model_dir: str
    seeds: tuple[int, ...]
    calibration_path: str | None
    stiffness: list[SweepResult]
    sensing: list[SweepResult]
    sensing_calibrated: list[SweepResult]
    speed: list[SweepResult]
    calibrated: list[SweepResult]
    decision_condition: str
    decision: TransferDecision
    correction: CorrectionVerdict
    working_range: tuple[float, float] | None
    measured_kp_scales: dict[str, float]
    sag_comparison: list[JointSagComparison]

    def to_json(self) -> str:
        """`report.json` contents."""
        return json.dumps(asdict(self), indent=2)


def write_plots(report: RobustnessReport, folder: Path) -> list[Path]:
    """Stiffness sweep and saturation plots; returns the written paths."""
    folder.mkdir(parents=True, exist_ok=True)
    kp_path, sat_path = folder / KP_SWEEP_PNG, folder / SATURATION_PNG
    plot_kp_sweep(report.stiffness, report.measured_kp_scales, kp_path)
    plot_saturation(report.stiffness, sat_path)
    return [kp_path, sat_path]


def robustness_readme(report: RobustnessReport) -> str:
    """Markdown answering: feedback or feedforward, working stiffness range, transfer or retrain."""
    sections = [
        _header(report),
        _verdict(report),
        _correction(report.correction),
        _range(report),
        _table_section("Stiffness sweep (every actuator kp scaled)", report.stiffness),
        _table_section("Sensing and timing (encoder rounding, delay in control steps)", report.sensing),
        _table_section("Sensing and timing in the calibrated sim", report.sensing_calibrated),
        _table_section("Reduced command speed", report.speed),
        _table_section("Calibrated sim, four controllers", report.calibrated),
        _sag_section(report.sag_comparison),
    ]
    return "\n\n".join(s for s in sections if s) + "\n"


def _header(report: RobustnessReport) -> str:
    """Title, inputs, and the provisional caveat when no calibration exists."""
    lines = [
        "# Phase 2: sim robustness of the reach policy",
        "",
        f"Checkpoint `{report.model_dir}`, seeds {list(report.seeds)}. Errors are holding-window means (mm),",
        "p95 in parentheses; baselines are deterministic and evaluated once.",
    ]
    if report.calibration_path is None:
        lines += ["", "**Provisional:** no Phase 0 calibration was given, so the decision uses the placeholder MJCF."]
    else:
        lines += ["", f"Calibration: `{report.calibration_path}`."]
    return "\n".join(lines)


def _verdict(report: RobustnessReport) -> str:
    """Transfer-or-retrain verdict with one line per check."""
    word = "TRANSFER" if report.decision.transfer else "RETRAIN"
    reasons = "\n".join(f"- {r}" for r in report.decision.reasons)
    return f"## Verdict: {word}\n\nCondition: {report.decision_condition}.\n\n{reasons}"


def _correction(verdict: CorrectionVerdict) -> str:
    """Feedback-or-feedforward answer with the growth factors behind it."""
    position = "n/a" if verdict.position is None else f"{verdict.position:.2f}"
    return (
        f"## Feedback or feedforward: {verdict.kind}\n\n"
        "Growth = worst error over the sweep / error at nominal kp. "
        f"Policy {verdict.trained_growth:.2f}x, integral control {verdict.integral_growth:.2f}x, "
        f"gravity compensation {verdict.gravity_comp_growth:.2f}x "
        f"(position {position} on a log scale: 0 = like integral, 1 = like gravity comp)."
    )


def _range(report: RobustnessReport) -> str:
    """Working stiffness range and the measured stiffness marks."""
    if report.working_range is None:
        text = "The policy misses the transfer error limit even at nominal stiffness."
    else:
        low, high = report.working_range
        text = f"The policy meets the transfer error limit for kp scales {low:g} to {high:g}."
    if report.measured_kp_scales:
        marks = ", ".join(f"{j} {s:.2f}x" for j, s in report.measured_kp_scales.items())
        text += f" Phase 0 measured stiffness: {marks}."
    else:
        text += " No Phase 0 stiffness measured yet."
    return f"## Working stiffness range\n\n{text}"


def _table_section(title: str, results: list[SweepResult]) -> str:
    """Pivot table: one row per parameter value, one column per controller."""
    if not results:
        return ""
    parameter = results[0].parameter
    values = sorted({r.value for r in results})
    header = f"| {parameter} | " + " | ".join(CONTROLLER_ORDER) + " | policy saturation |"
    rows = [header, "|" + "---|" * (len(CONTROLLER_ORDER) + 2)]
    rows += [_pivot_row(results, v) for v in values]
    return f"## {title}\n\n" + "\n".join(rows)


def _pivot_row(results: list[SweepResult], value: float) -> str:
    """One pivot row."""
    cells = {r.controller: r for r in results if r.value == value}
    errors = [f"{cells[c].mean_mm:.2f} ({cells[c].p95_mm:.2f})" if c in cells else "-" for c in CONTROLLER_ORDER]
    sat = cells.get(TRAINED)
    sat_text = "-" if sat is None or sat.saturation_fraction is None else f"{sat.saturation_fraction:.1%}"
    return f"| {value:g} | " + " | ".join(errors) + f" | {sat_text} |"


def _sag_section(rows: list[JointSagComparison]) -> str:
    """Calibrated sim against Phase 0, joint by joint."""
    if not rows:
        return ""
    table = ["| joint | real mean sag (deg) | sim mean sag (deg) |", "|---|---|---|"]
    table += [f"| {r.joint} | {r.real_mean_sag_deg:.2f} | {r.sim_mean_sag_deg:.2f} |" for r in rows]
    return "## Calibrated sim against Phase 0 sag\n\n" + "\n".join(table)

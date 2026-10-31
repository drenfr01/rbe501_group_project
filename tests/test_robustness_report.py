"""Phase 2 report: JSON serialization, README text, and plots from hand-built results."""

import json

from rbe_501_so101.config import RobustnessConfig
from rbe_501_so101.rl.robustness import TRAINED, SweepResult, classify_correction, transfer_decision
from rbe_501_so101.rl.robustness_report import RobustnessReport, robustness_readme, write_plots


def _result(controller: str, parameter: str, value: float, mean_mm: float, sat: float | None = None) -> SweepResult:
    """Hand-built sweep point."""
    return SweepResult(controller, parameter, value, mean_mm, 2 * mean_mm, {"0": mean_mm}, sat)


def _stiffness() -> list[SweepResult]:
    """Integral flat, gravity comp growing, policy flat."""
    rows = []
    for scale, grav in ((0.5, 10.0), (1.0, 0.7), (2.0, 5.0)):
        rows += [
            _result("integral", "kp_scale", scale, 0.7),
            _result("gravity_comp", "kp_scale", scale, grav),
            _result("ik_hold", "kp_scale", scale, 10.0 / scale),
            _result(TRAINED, "kp_scale", scale, 0.4, 0.01),
        ]
    return rows


def _report() -> RobustnessReport:
    """Report with only a stiffness sweep and a provisional decision."""
    stiffness = _stiffness()
    decision_rows = [r for r in stiffness if r.value == 1.0]
    return RobustnessReport(
        model_dir="checkpoints/x",
        seeds=(0,),
        calibration_path=None,
        stiffness=stiffness,
        sensing=[],
        sensing_calibrated=[],
        speed=[],
        calibrated=[],
        decision_condition="nominal sim + encoder rounding + 1-step delay",
        decision=transfer_decision(decision_rows, RobustnessConfig()),
        correction=classify_correction(stiffness),
        working_range=(0.5, 2.0),
        measured_kp_scales={"shoulder_lift": 0.6},
        sag_comparison=[],
    )


def test_report_serializes_to_json():
    data = json.loads(_report().to_json())
    assert data["correction"]["kind"] == "feedback"
    assert len(data["stiffness"]) == 12


def test_readme_answers_the_three_questions():
    text = robustness_readme(_report())
    assert "feedback" in text
    assert "0.5" in text and "2" in text
    assert "Verdict" in text and "TRANSFER" in text
    assert "provisional" in text.lower()


def test_plots_are_written(tmp_path):
    paths = write_plots(_report(), tmp_path)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)

"""Summarize both frozen stages; preserve nominal regressions in the final decision."""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def summarize(rows: list[dict]) -> dict:
    """Report task outcome, closure diagnostics and all sampled safety events."""
    closed = [r for r in rows if r["close_started"]]
    keys = ("closure_position_error_mm", "closure_axis_error_deg", "ik_position_error_mm", "ik_axis_error_deg")
    return {"episodes": len(rows), "successes": sum(r["task"]["final_success"] for r in rows),
            "acquisition_aborts": sum(r["acquisition_abort"] is not None for r in rows),
            "table_contact_episodes": sum(r["task"]["robot_table_contact_steps"] > 0 for r in rows),
            "approach_contact_episodes": sum(r["task"]["approach_contact_steps"] > 0 for r in rows),
            "warning_episodes": sum(r["task"]["warnings"] for r in rows),
            "closed_diagnostics": {k: {"mean": sum(r["diagnostics"][k] for r in closed) / len(closed),
                                       "maximum": max(r["diagnostics"][k] for r in closed)}
                                   for k in keys if closed and all(r["diagnostics"][k] is not None for r in closed)}}


def run() -> None:
    """Require nominal-grid preservation in addition to the independent validation gate."""
    development = json.loads((ROOT / "development/episodes.json").read_text())
    validation = json.loads((ROOT / "validation/episodes.json").read_text())
    grid = [r for r in development if r["case"]["radius_m"] is not None]
    assessment = json.loads((ROOT / "assessment.json").read_text())
    result = {"development_workspace": summarize(grid),
              "development_yaw": summarize([r for r in development if r["case"]["radius_m"] is None]),
              "validation": {name: summarize([r for r in validation if r["controller"] == name]) for name in ("scripted", "oracle")},
              "preserves_development_workspace": all(r["task"]["final_success"] for r in grid),
              "promote_controller": False, "start_training": False,
              "reason": "Improves rotation handling, but fails nominal workspace preservation and independent nominal/per-yaw gates.",
              "independent_passed_validation": assessment["passed_validation"]}
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    run()

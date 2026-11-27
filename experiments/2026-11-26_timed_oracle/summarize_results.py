"""Summarize the frozen timing repair without excluding failed cases."""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def summarize(rows: list[dict]) -> dict:
    """Keep success and contact denominators alongside conditional closure diagnostics."""
    closed = [r for r in rows if r["close_started"]]
    keys = ("closure_position_error_mm", "closure_axis_error_deg", "ik_position_error_mm", "ik_axis_error_deg")
    return {"episodes": len(rows), "successes": sum(r["task"]["final_success"] for r in rows),
            "table_contact_episodes": sum(r["task"]["robot_table_contact_steps"] > 0 for r in rows),
            "approach_contact_episodes": sum(r["task"]["approach_contact_steps"] > 0 for r in rows),
            "warning_episodes": sum(r["task"]["warnings"] for r in rows),
            "closed_diagnostics": {k: {"mean": sum(r["diagnostics"][k] for r in closed) / len(closed),
                                       "maximum": max(r["diagnostics"][k] for r in closed)}
                                   for k in keys if closed and all(r["diagnostics"][k] is not None for r in closed)}}


def run() -> None:
    """Save both nominal and rotated results; the independent gate remains in assessment.json."""
    recorded = json.loads((ROOT / "recorded/episodes.json").read_text())
    validation = json.loads((ROOT / "validation/episodes.json").read_text())
    result = {"recorded": summarize(recorded),
              "recorded_grid": summarize([r for r in recorded if r["case"]["radius_m"] is not None]),
              "recorded_yaw": summarize([r for r in recorded if r["case"]["radius_m"] is None]),
              "validation": {c: summarize([r for r in validation if r["controller"] == c]) for c in ("scripted", "timed")}}
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    run()

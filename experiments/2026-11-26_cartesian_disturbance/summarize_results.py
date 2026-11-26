"""Summarize every recorded condition, including failed or ineligible acquisitions."""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def summarize(rows: list[dict]) -> dict:
    """Report descriptive physical metrics separately from episode denominators."""
    eligible = [r for r in rows if r["eligible"]]
    metrics = ("cube_rms_mm", "pinch_rms_mm", "peak_cube_mm", "peak_slip_mm")
    return {"episodes": len(rows), "eligible": len(eligible),
            "final_successes": sum(r["success"] for r in rows),
            "transient_or_final_success_loss": sum(r["lost_success"] for r in rows),
            "two_pad_grasp_loss": sum(r["lost_grasp"] for r in rows),
            "table_contact_episodes": sum(r["table_steps"] > 0 for r in rows),
            "approach_contact_episodes": sum(r["approach_contact"] for r in rows),
            "warning_episodes": sum(r["warning"] for r in rows),
            "stable_recovery_confirmed": sum(r["recovery_steps"] is not None for r in eligible),
            "max_recovery_confirmation_steps": max((r["recovery_steps"] or 0 for r in eligible), default=0),
            "eligible_descriptive_means": {k: sum(r[k] for r in eligible) / len(eligible) for k in metrics} if eligible else {},
            "max_peak_cube_mm": max((r["peak_cube_mm"] for r in eligible), default=None)}


def run() -> None:
    """Persist grouped and overall summaries from the final complete episode file."""
    rows = json.loads((ROOT / "episodes.json").read_text())
    result = {"overall": {c: summarize([r for r in rows if r["controller"] == c]) for c in ("scripted", "sac")},
              "forced": {c: summarize([r for r in rows if r["controller"] == c and r["pulse"] != "none"]) for c in ("scripted", "sac")},
              "conditions": [{"controller": c, "pulse": p, **summarize([r for r in rows if r["controller"] == c and r["pulse"] == p])}
                             for p in dict.fromkeys(r["pulse"] for r in rows) for c in ("scripted", "sac")]}
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "conditions"}, indent=2))


if __name__ == "__main__":
    run()

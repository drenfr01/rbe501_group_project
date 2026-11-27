"""Summarize all full-task carry-push outcomes and the separate matched-path diagnostics."""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def summarize(rows: list[dict]) -> dict:
    """Keep failure denominators intact when reporting complete placement and final error."""
    return {"episodes": len(rows), "eligible": sum(r["eligible"] for r in rows),
            "successes": sum(r["success"] for r in rows),
            "transport_grasp_losses": sum(r["transport_grasp_loss"] for r in rows),
            "table_contact_episodes": sum(r["table_steps"] > 0 for r in rows),
            "warning_episodes": sum(r["warning"] for r in rows),
            "mean_final_error_mm": sum(r["final_error_mm"] for r in rows) / len(rows),
            "max_final_error_mm": max(r["final_error_mm"] for r in rows),
            "residual_steps": sum(r["residual_steps"] for r in rows),
            "failures": [{k: r[k] for k in ("seed", "mass_kg", "pulse", "eligible", "failure_reason")}
                         for r in rows if not r["success"]]}


def path_summary(pairs: list[dict], name: str) -> dict:
    """Path diagnostics exclude only unobserved windows, never complete-task failures."""
    paths = [p[name] for p in pairs if p["common_eligible"] and p[name] is not None]
    recovered = [p["confirmed_recovery_periods"] for p in paths if p["confirmed_recovery_periods"] is not None]
    return {"complete_windows": len(paths), "unobserved_windows": len(pairs) - len(paths),
            "mean_path_rms_mm": sum(p["path_rms_mm"] for p in paths) / len(paths) if paths else None,
            "mean_peak_grip_relative_mm": sum(p["peak_grip_relative_deviation_mm"] for p in paths) / len(paths) if paths else None,
            "recovery_confirmed": len(recovered), "max_confirmation_periods": max(recovered, default=None)}


def run() -> None:
    """Persist descriptive summaries from the complete immutable evaluation files."""
    rows = json.loads((ROOT / "episodes.json").read_text())
    pairs = json.loads((ROOT / "paired_results.json").read_text())
    result = {"overall": {c: summarize([r for r in rows if r["controller"] == c]) for c in ("scripted", "sac")},
              "forced": {c: summarize([r for r in rows if r["controller"] == c and r["pulse"] != "none"]) for c in ("scripted", "sac")},
              "unforced": {c: summarize([r for r in rows if r["controller"] == c and r["pulse"] == "none"]) for c in ("scripted", "sac")},
              "paths": {c: path_summary(pairs, key) for c, key in (("scripted", "script_path"), ("sac", "rl_path"))},
              "conditions": [{"controller": c, "pulse": p, **summarize([r for r in rows if r["controller"] == c and r["pulse"] == p])}
                             for p in dict.fromkeys(r["pulse"] for r in rows) for c in ("scripted", "sac")]}
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "conditions"}, indent=2))


if __name__ == "__main__":
    run()

"""Visualize paired full-task outcomes without treating matched success as RL superiority."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def plot() -> None:
    """Show task completion and descriptive paired placement errors on common conditions."""
    baseline = json.loads((ROOT / "scripted_eval.json").read_text())
    learned = json.loads((ROOT / "learned_eval.json").read_text())
    keyed = {(r["seed"], r["mass_kg"], r["yaw_offset_deg"]): r for r in learned}
    x = [r["error_mm"] for r in baseline]
    y = [keyed[(r["seed"], r["mass_kg"], r["yaw_offset_deg"])]["error_mm"] for r in baseline]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    successes = [sum(r["success"] for r in rows) for rows in (baseline, learned)]
    left.bar(("Scripted baseline", "Fresh SAC residual"), successes, color=("tab:blue", "tab:orange"))
    left.set(title="Complete released placements", ylabel="Successful trials", ylim=(0, 65))
    for index, count in enumerate(successes):
        left.text(index, count - 2, f"{count}/60", color="white", ha="center", va="top")
    right.scatter(x, y, s=24, alpha=.7)
    right.plot((0, 8), (0, 8), "k--", alpha=.5)
    right.set(title="Paired final cube XY errors", xlabel="Scripted error (mm)", ylabel="SAC residual error (mm)", xlim=(0, 8), ylim=(0, 8))
    right.grid(alpha=.2)
    fig.suptitle("New full-task environment: short CUDA SAC stability pilot")
    fig.text(.5, .015, "5 placement centers × 4 masses × 3 yaws; one training seed; no RL-benefit claim", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(ROOT / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot()

"""Plot fixed-window versus bounded-settling recorded results and fresh yaw coverage."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def plot() -> None:
    """Distinguish timing-only recorded comparisons from full-controller fresh comparisons."""
    rows = json.loads((ROOT / "validation/episodes.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    angles = (-45, -30, -15, 0, 15, 30, 45)
    for name, label in (("scripted", "Original radial script"), ("timed", "Orientation + bounded timing")):
        rates = []
        for angle in angles:
            group = [r for r in rows if r["controller"] == name and r["case"]["yaw_offset_deg"] == angle]
            rates.append(100 * sum(r["task"]["final_success"] for r in group) / len(group))
        left.plot(angles, rates, marker="o", label=label)
    left.set(title="Independent yaw validation", xlabel="Yaw offset (degrees)", ylabel="Successful lifts (%)", xticks=angles, ylim=(0, 105))
    left.legend(fontsize=8)
    x = [0, 1]
    right.bar([v - .18 for v in x], (152 / 180 * 100, 268 / 280 * 100), width=.36, label="Previous oracle")
    right.bar([v + .18 for v in x], (100, 100), width=.36, label="Bounded timing")
    right.set(title="Identical recorded cases: timing change", ylabel="Successful lifts (%)", xticks=x,
              xticklabels=("Workspace grid\n180 cases", "Yaw cases\n280 cases"), ylim=(0, 105))
    right.legend(loc="lower left", fontsize=8)
    fig.suptitle("Bounded settling repairs the tested acquisition regressions")
    fig.text(.5, .012, "Fresh yaw points: 10 placements × 4 masses; original 200-step task; no RL training", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(ROOT / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot()

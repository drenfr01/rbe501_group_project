"""Plot frozen orientation gains alongside nominal-grid regressions."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def plot() -> None:
    """Keep success rates on a full scale; show both stages and original nominal coverage."""
    rows = json.loads((ROOT / "validation/episodes.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    angles = (-45, -30, -15, 0, 15, 30, 45)
    for name, label in (("scripted", "Original script"), ("oracle", "Oracle prototype")):
        rates = []
        for angle in angles:
            group = [r for r in rows if r["controller"] == name and r["case"]["yaw_offset_deg"] == angle]
            rates.append(100 * sum(r["task"]["final_success"] for r in group) / len(group))
        left.plot(angles, rates, marker="o", label=label)
    left.set(title="Independent yaw validation", xlabel="Yaw offset (degrees)", ylabel="Successful lifts (%)", xticks=angles, ylim=(0, 105))
    left.legend()
    right.bar(("Original script", "Oracle prototype"), (100, 152 / 180 * 100), color=("tab:blue", "tab:orange"))
    right.set(title="Recorded nominal workspace grid", ylabel="Successful lifts (%)", ylim=(0, 105))
    right.text(0, 97, "180/180", ha="center", va="top", color="white")
    right.text(1, 81, "152/180", ha="center", va="top", color="white")
    fig.suptitle("Orientation helps, but the prototype fails the no-regression gate")
    fig.text(.5, .012, "Yaw points: 10 placements × 4 masses; grid: 45 positions × 4 masses; no training", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(ROOT / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot()

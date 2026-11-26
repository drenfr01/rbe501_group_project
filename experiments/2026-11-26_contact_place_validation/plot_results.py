"""Show the last-timeout repair without hiding the rejected intermediate candidate."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def plot() -> None:
    """Keep matched old cases separate from the new independent complete-task matrix."""
    rows = json.loads((ROOT / "episodes.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.bar(("Previous support", "Contact-aware"), (59 / 60 * 100, 100), color=("tab:blue", "tab:orange"))
    left.set(title="Same 60 last-validation conditions", ylabel="Complete sequences (%)", ylim=(0, 105))
    left.text(0, 94, "59/60", ha="center", va="top", color="white")
    left.text(1, 96, "60/60", ha="center", va="top", color="white")
    masses, angles = (.005, .01, .015, .02), (-30, 0, 30)
    matrix = [[sum(r["complete_success"] for r in rows if r["mass_kg"] == mass and r["yaw_offset_deg"] == yaw)
               for yaw in angles] for mass in masses]
    right.imshow(matrix, vmin=0, vmax=10, cmap="Blues", aspect="auto")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            right.text(j, i, f"{value}/10", ha="center", va="center", color="white")
    right.set(title="Fresh validation: 120/120", xlabel="Cube yaw offset (degrees)", ylabel="Cube mass (g)",
              xticks=range(3), xticklabels=angles, yticks=range(4), yticklabels=(5, 10, 15, 20))
    fig.suptitle("Contact-aware lowering repairs the final placement timeout")
    fig.text(.5, .015, "All 240 prior conditions retained; fresh points repeat ten placements; no RL training", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(ROOT / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot()

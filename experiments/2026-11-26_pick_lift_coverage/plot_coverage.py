"""Plot frozen yaw coverage, separating exploratory and independent placements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def plot() -> None:
    """Aggregate repeated mass trials descriptively without treating them as independent."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    angles = (-45, -30, -15, 0, 15, 30, 45)
    for ax, folder, title in zip(axes, (ROOT, ROOT / "confirmation"), ("Initial placements", "Independent placements")):
        rows = json.loads((folder / "episodes.json").read_text())
        for name, label in (("scripted", "Scripted"), ("sac", "Frozen SAC")):
            rates = []
            for angle in angles:
                group = [r for r in rows if r["controller"] == name and r["case"]["radius_m"] is None
                         and r["case"]["yaw_offset_deg"] == angle]
                rates.append(100 * sum(r["task"]["final_success"] for r in group) / len(group))
            ax.plot(angles, rates, marker="s" if name == "scripted" else "o",
                    markersize=8 if name == "scripted" else 4,
                    linestyle="--" if name == "scripted" else "-", label=label, alpha=.8)
        ax.set(title=title, xlabel="Cube yaw offset from radial alignment (degrees)",
               ylabel="Successful lifts (%)", ylim=(-3, 103), xticks=angles)
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle("Acquisition weakness: independent cube orientation")
    fig.text(.5, .015, "Each point: 10 placements × 4 masses; ±45° are equivalent square orientations", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(ROOT / "yaw_coverage.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot()

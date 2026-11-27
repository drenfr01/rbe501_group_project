"""Render raw paired hold-retention rates; run after the complete frozen sweep."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def plot() -> None:
    """Show four directions separately without treating repeated masses as independent."""
    rows = json.loads((ROOT / "episodes.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, direction in zip(axes.flat, ("+x", "-x", "+y", "-y")):
        for name, label in (("scripted", "Scripted"), ("sac", "Frozen SAC")):
            rates = []
            for force in (0, .1, .2, .4, .8):
                pulse = "none" if force == 0 else f"{force:g}N_{direction}"
                group = [r for r in rows if r["controller"] == name and r["pulse"] == pulse and r["eligible"]]
                rates.append(100 * sum(r["success"] for r in group) / len(group) if group else float("nan"))
            ax.plot((0, .1, .2, .4, .8), rates, marker="s" if name == "scripted" else "o",
                    markersize=8 if name == "scripted" else 4,
                    linestyle="--" if name == "scripted" else "-", label=label, alpha=.8)
        ax.set(title=f"World {direction}", xlabel="Applied force (N)", ylabel="Final successful holds (%)", ylim=(-3, 103))
        ax.grid(alpha=.25)
    axes[0, 0].legend()
    fig.suptitle("99 ms cube pushes: frozen SAC versus scripted lift")
    fig.text(.5, .015, "10 paired placements × 4 masses per point; one trained seed; nominal arm effects", ha="center")
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(ROOT / "retention.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot()

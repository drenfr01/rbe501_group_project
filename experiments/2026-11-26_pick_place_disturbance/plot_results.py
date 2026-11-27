"""Display complete-task retention and cluster-level adjusted placement-error differences."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def plot() -> None:
    """Separate practical benefit thresholds from descriptive pooled motion differences."""
    rows = json.loads((ROOT / "episodes.json").read_text())
    pairs = json.loads((ROOT / "paired_results.json").read_text())
    assessment = json.loads((ROOT / "assessment.json").read_text())
    names = list(dict.fromkeys(r["pulse"] for r in rows))
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    for controller, shift, label in (("scripted", -.18, "Scripted"), ("sac", .18, "Frozen SAC")):
        rates = []
        for pulse in names:
            group = [r for r in rows if r["controller"] == controller and r["pulse"] == pulse]
            rates.append(100 * sum(r["success"] for r in group) / len(group))
        left.bar([i + shift for i in range(len(names))], rates, width=.36, label=label)
    left.set(title="Complete placements after carrying pushes", ylabel="Success (%)", ylim=(0, 105),
             xticks=range(len(names)), xticklabels=[n.replace("N_", "\n") for n in names])
    left.tick_params(axis="x", labelsize=8)
    left.set_xlabel("Pulse magnitude (N) and world direction")
    left.legend(loc="lower left")
    seeds = sorted({p["seed"] for p in pairs if p["common_eligible"]})
    means = [sum(p["adjusted_final_error_difference_mm"] for p in pairs if p["common_eligible"] and p["seed"] == s)
             / sum(p["common_eligible"] and p["seed"] == s for p in pairs) for s in seeds]
    right.scatter(range(len(seeds)), means)
    right.axhline(0, color="black", linestyle="--", alpha=.5)
    right.axhline(-.5, color="green", linestyle=":", label="Predeclared 0.5mm improvement")
    metric = assessment["adjusted_final_error_difference_mm"]
    right.set(title=f"RL − script adjusted error: mean {metric['mean']:.3f} mm",
              xlabel="Placement cluster", ylabel="Forced-minus-own-control difference (mm)")
    right.grid(alpha=.2)
    right.legend(fontsize=8)
    fig.suptitle("Full-task carry robustness: frozen controllers, no retraining")
    fig.text(.5, .015, "10 placements repeated over masses/forces; nominal yaw; one trained policy seed", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(ROOT / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot()

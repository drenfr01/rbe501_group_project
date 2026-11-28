"""Plot success rate and lift height across object masses."""

from pathlib import Path

import matplotlib
from rbe_501_so101.config import PickLiftTaskConfig
from rbe_501_so101.rl.pick_lift import MassResult

matplotlib.use("Agg")

M_TO_CM = 100.0
PERCENT = 100.0


def plot_mass_sweep(report: dict[str, list[MassResult]], path: Path) -> None:
    """Plot success rate and mean final lift against cube mass for each policy."""
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, results in report.items():
        masses = [result.mass_kg for result in results]
        top.plot(masses, [PERCENT * result.success_rate for result in results], marker="o", label=name)
        bottom.plot(masses, [M_TO_CM * result.mean_lift_m for result in results], marker="o", label=name)
    bottom.axhline(M_TO_CM * PickLiftTaskConfig().lift_success_m, color="red", linestyle="--", label="success height")
    top.set_ylabel("success rate (%)")
    top.set_title("Pick-and-lift vs. object mass")
    bottom.set_xlabel("object mass (kg)")
    bottom.set_ylabel("final lift height (cm)")
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
        ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

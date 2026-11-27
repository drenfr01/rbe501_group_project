"""Rebuild the known-case closing diagnostic figure from the preserved raw traces."""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="so101_plot_"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot() -> None:
    """Distinguish fixed arm commands from contact-induced physical deflection."""
    root = Path(__file__).parent
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    profiles = (("position_minus10", "Target -10 deg", "#ba332f"),
                ("position_zero", "Target 0 deg", "#237e48"),
                ("contact_stop1", "First-contact stop", "#3269ab"))
    for name, label, color in profiles:
        data = json.loads((root / "diagnostic_traces" / f"{name}_effects_True.json").read_text())
        frames = [r for r in data["trace"] if r["stage"] == "closing"]
        t = np.arange(1, len(frames) + 1) * .033
        values = ([r["jaw_command_deg"] for r in frames],
                  [min(r["fixed"]["normal_force_n"], r["moving"]["normal_force_n"]) for r in frames],
                  [r["cube_shift_xy_mm"] for r in frames], [r["pinch_drift_mm"] for r in frames])
        for ax, value in zip(axes.flat, values):
            ax.plot(t, value, label=label, color=color, linewidth=1.6)
    finish(fig, axes, root)


def finish(fig, axes, root: Path) -> None:
    """Label physical units and state that this is one diagnostic placement."""
    labels = ("Jaw command (deg)", "Min. of two pad normal forces (N)",
              "Cube horizontal displacement (mm)", "Physical pinch displacement (mm)")
    for ax, label in zip(axes.flat, labels):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    for ax in axes[1]:
        ax.set_xlabel("Time since closing start (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("5 g / placement 68006 / arm effects: fixed arm commands, not frozen joints", fontsize=11)
    fig.tight_layout()
    fig.savefig(root / "closing_diagnostic.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    plot()

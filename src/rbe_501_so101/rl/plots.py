"""Matplotlib figures for experiment records."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rbe_501_so101.config import TrainConfig  # noqa: E402
from rbe_501_so101.rl.reporting import TrainingCurve  # noqa: E402

M_TO_MM = 1000.0
RAW_ALPHA = 0.15
FIG_SIZE = (9, 5)
DPI = 150


def plot_training_reward(curves: list[TrainingCurve], path: Path) -> None:
    """Episode reward vs. env steps per seed: faint raw values plus a rolling mean."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for curve in curves:
        (line,) = ax.plot(curve.timesteps, curve.smoothed(), label=f"seed {curve.seed}")
        ax.plot(curve.timesteps, curve.rewards, color=line.get_color(), alpha=RAW_ALPHA, linewidth=0.5)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode reward (150 steps)")
    ax.set_title("SAC training reward (rolling mean of 50 episodes)")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_holding_error(rows: dict[str, dict], cfg: TrainConfig, path: Path) -> None:
    """Mean and p95 holding error per policy, with per-seed means and the pass thresholds."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    names = list(rows)
    xs = range(len(names))
    width = 0.38
    ax.bar([x - width / 2 for x in xs], [rows[n]["mean_mm"] for n in names], width, label="mean")
    ax.bar([x + width / 2 for x in xs], [rows[n]["p95_mm"] for n in names], width, label="p95")
    _plot_seed_means(ax, rows, names, width)
    ax.axhline(M_TO_MM * cfg.success_mean_m, color="green", linestyle="--", label="mean target")
    ax.axhline(M_TO_MM * cfg.success_p95_m, color="red", linestyle="--", label="p95 target")
    ax.set_xticks(list(xs), names, rotation=15)
    ax.set_ylabel("holding error, last 30 steps (mm)")
    ax.set_title("Holding error vs. targets (random baseline omitted: ~345 mm)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    _save(fig, path)


def _plot_seed_means(ax: plt.Axes, rows: dict[str, dict], names: list[str], width: float) -> None:
    """Overlay each seed's mean error as black dots on the mean bar."""
    for x, name in enumerate(names):
        seed_means = list(rows[name]["per_seed_mean_mm"].values())
        if len(seed_means) > 1:
            ax.scatter([x - width / 2] * len(seed_means), seed_means, color="black", s=12, zorder=3)


def _save(fig: plt.Figure, path: Path) -> None:
    """Write a figure to disk and release it."""
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)

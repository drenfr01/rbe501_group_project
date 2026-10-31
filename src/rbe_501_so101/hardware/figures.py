"""Report figures for the hardware measurements: friction bands, direction errors, stiffness, gain sweep."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rbe_501_so101.hardware.direction_errors import DirectionalError  # noqa: E402
from rbe_501_so101.hardware.sag import JointComplianceFit  # noqa: E402

FIG_SIZE = (9, 5)
WIDE_FIG_SIZE = (11, 4.5)
DPI = 150
BAR_WIDTH = 0.8
GRID_ALPHA = 0.3
FIT_POINTS = 50
WITH_MARKER, AGAINST_MARKER = "v", "^"


@dataclass(frozen=True)
class BandSeries:
    """Mean friction band per joint for one procedure, with an optional (low, high) range."""

    label: str
    mean_deg: Mapping[str, float]
    range_deg: Mapping[str, tuple[float, float]] = field(default_factory=dict)


def plot_friction_bands(series: Sequence[BandSeries], joints: Sequence[str], path: Path) -> None:
    """Grouped bars: full friction band per joint, one bar per measuring procedure."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    width = BAR_WIDTH / len(series)
    for k, s in enumerate(series):
        xs = np.arange(len(joints)) + (k - (len(series) - 1) / 2) * width
        means = [s.mean_deg.get(j, np.nan) for j in joints]
        ax.bar(xs, means, width, label=s.label, yerr=_range_errors(s, joints, means), capsize=4)
    ax.set_xticks(range(len(joints)), joints)
    ax.set_ylabel("against-gravity minus with-gravity error (deg)")
    ax.set_title("Friction band: how much the result depends on the approach direction")
    _finish(ax, fig, path)


def _range_errors(series: BandSeries, joints: Sequence[str], means: list[float]) -> np.ndarray | None:
    """Asymmetric error bars from the (low, high) range, or None when there is none."""
    if not series.range_deg:
        return None
    low = [m - series.range_deg.get(j, (m, m))[0] for j, m in zip(joints, means, strict=True)]
    high = [series.range_deg.get(j, (m, m))[1] - m for j, m in zip(joints, means, strict=True)]
    return np.array([low, high])


def plot_direction_errors(
    sessions: Mapping[str, Sequence[DirectionalError]],
    kp_nm_per_rad: Mapping[str, float],
    joints: Sequence[str],
    path: Path,
) -> None:
    """Error vs gravity load per joint: with-gravity (down markers), against-gravity (up), spring sag line."""
    fig, axes = plt.subplots(1, len(joints), figsize=WIDE_FIG_SIZE, squeeze=False)
    for ax, joint in zip(axes[0], joints, strict=True):
        for label, errors in sessions.items():
            _scatter_joint(ax, label, [e for e in errors if e.joint == joint])
        _spring_line(ax, joint, kp_nm_per_rad, sessions)
        ax.set_title(joint)
        ax.set_xlabel("gravity load (N·m)")
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("error along gravity (deg)")
    fig.suptitle("Where the joint stops: arriving with gravity vs against it")
    _finish(axes[0][-1], fig, path)


def _scatter_joint(ax: plt.Axes, label: str, errors: Sequence[DirectionalError]) -> None:
    """With- and against-gravity points of one session in one colour."""
    load = [e.gravity_load_nm for e in errors]
    with_deg = [e.with_gravity_deg for e in errors]
    (with_pts,) = ax.plot(load, with_deg, WITH_MARKER, linestyle="", label=f"{label}, with gravity")
    against_deg = [e.against_gravity_deg for e in errors]
    ax.plot(load, against_deg, AGAINST_MARKER, linestyle="", color=with_pts.get_color(), label=f"{label}, against")


def _spring_line(
    ax: plt.Axes, joint: str, kp: Mapping[str, float], sessions: Mapping[str, Sequence[DirectionalError]]
) -> None:
    """Spring sag `load / kp` over the plotted load range, when the joint has a fitted kp."""
    loads = [e.gravity_load_nm for errors in sessions.values() for e in errors if e.joint == joint]
    if joint not in kp or not loads:
        return
    x = np.linspace(0.0, max(loads), FIT_POINTS)
    ax.plot(x, np.rad2deg(x / kp[joint]), "k--", linewidth=1, label=f"spring sag, kp {kp[joint]:.1f}")


def plot_stiffness(fits: Sequence[JointComplianceFit], mjcf_kp: Mapping[str, float], path: Path) -> None:
    """MJCF kp vs fitted real kp (95% interval) per identifiable joint."""
    shown = [f for f in fits if f.identifiable and f.kp_nm_per_rad is not None]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    xs = np.arange(len(shown))
    width = BAR_WIDTH / 2
    ax.bar(xs - width / 2, [mjcf_kp.get(f.joint, np.nan) for f in shown], width, label="sim model (MJCF)")
    real = [f.kp_nm_per_rad for f in shown]
    low = [r - (f.kp_ci_low or r) for r, f in zip(real, shown, strict=True)]
    high = [_finite(f.kp_ci_high, r) - r for r, f in zip(real, shown, strict=True)]
    err = [low, high]
    ax.bar(xs + width / 2, real, width, yerr=err, capsize=4, label="real arm, P = 24 (95% interval)")
    ax.set_xticks(xs, [f.joint for f in shown])
    ax.set_ylabel("stiffness kp (N·m/rad)")
    ax.set_title("Joint stiffness: simulator vs measured")
    _finish(ax, fig, path)


def _finite(value: float | None, fallback: float) -> float:
    """`value` unless it is missing or infinite."""
    return fallback if value is None or not np.isfinite(value) else value


def plot_gain_sweep(gains: Sequence[int], errors_deg: Mapping[str, Sequence[float]], path: Path) -> None:
    """Largest miss per joint vs servo P gain, with the best `c / P` curve for each."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    p = np.asarray(gains, dtype=float)
    p_fine = np.linspace(p.min(), p.max(), FIT_POINTS)
    for joint, err in errors_deg.items():
        (line,) = ax.plot(p, err, "o", label=joint)
        c = float(np.mean(np.asarray(err) * p))
        ax.plot(p_fine, c / p_fine, ":", color=line.get_color())
    ax.set_xticks(p)
    ax.set_xlabel("servo P gain (LeRobot default 16, Feetech factory 32)")
    ax.set_ylabel("largest miss over the visits (deg)")
    ax.set_title("Every error shrinks as 1 / P (dotted: c / P fits)")
    _finish(ax, fig, path)


def _finish(ax: plt.Axes, fig: plt.Figure, path: Path) -> None:
    """Grid, legend, save, close."""
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)

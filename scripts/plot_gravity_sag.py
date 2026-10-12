"""Plot how a position-controlled joint settles short of its command under gravity.

HOW TO RUN (from the project folder):
    uv run python scripts/plot_gravity_sag.py
    uv run python scripts/plot_gravity_sag.py --joint elbow_flex --start-deg 0 --target-deg 45

BEHAVIOR / OUTPUT:
    Starts from the home pose with one joint at --start-deg, lets it settle under
    gravity, then jumps the command to --target-deg and records every 1 ms physics
    step. Plots commanded vs. actual angle, in full and zoomed on the end.
    Panel titles flag runs where the arm ends resting on the table.
    Saves docs/figures/gravity_sag.png (override with --out).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from rbe_501_so101.config import SimConfig  # noqa: E402
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend  # noqa: E402

DEFAULT_OUT = Path("docs/figures/gravity_sag.png")
DEFAULT_JOINT = "shoulder_lift"
DEFAULT_START_DEG = 0.0
DEFAULT_TARGET_DEG = -30.0
RECORD_S = 1.0
PRE_SETTLE_S = 2.0
ZOOM_MARGIN_DEG = 1.5
FULL_MARGIN_DEG = 3.0
FIG_SIZE = (12, 4.8)
DPI = 150
S_TO_MS = 1000.0


@dataclass(frozen=True)
class Trace:
    """Commanded and actual angle of one joint over time, in degrees."""

    title: str
    t_ms: np.ndarray
    command_deg: float
    actual_deg: np.ndarray

    @property
    def settled_gap_deg(self) -> float:
        """Final command-minus-actual offset after the transient has died out."""
        return self.command_deg - float(self.actual_deg[-1])


@dataclass(frozen=True)
class Case:
    """One experiment: where the joint settles first and the command it then jumps to."""

    title: str
    start_rad: float
    command_rad: float


def place_joint(backend: MujocoSo101Backend, joint: int, angle_rad: float) -> None:
    """Reset to home and put one joint exactly at angle_rad with a matching command."""
    backend.reset(np.random.default_rng(0))
    model, data, ids = backend.model, backend.data, backend.ids
    data.qpos[ids.qpos_adr[joint]] = angle_rad
    data.ctrl[ids.actuator_ids[joint]] = angle_rad
    mujoco.mj_forward(model, data)


def run_physics(backend: MujocoSo101Backend, joint: int, seconds: float) -> np.ndarray:
    """Step the physics for `seconds` and return the joint angle at every step (radians)."""
    model, data = backend.model, backend.data
    qpos_adr = backend.ids.qpos_adr[joint]
    angles = [data.qpos[qpos_adr]]
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        angles.append(data.qpos[qpos_adr])
    return np.array(angles)


def simulate(backend: MujocoSo101Backend, joint: int, case: Case) -> Trace:
    """Settle at the start angle, then record the joint from the moment the new command is applied."""
    place_joint(backend, joint, case.start_rad)
    run_physics(backend, joint, PRE_SETTLE_S)
    backend.data.ctrl[backend.ids.actuator_ids[joint]] = case.command_rad
    actual = run_physics(backend, joint, RECORD_S)
    t_ms = np.arange(actual.size) * backend.model.opt.timestep * S_TO_MS
    title = f"{case.title}\n(ends resting on the table!)" if backend.data.ncon else case.title
    return Trace(title, t_ms, float(np.rad2deg(case.command_rad)), np.rad2deg(actual))


def plot_trace(ax: plt.Axes, trace: Trace, zoom: bool) -> None:
    """Draw one case; `zoom` frames the final gap, otherwise the whole motion is shown."""
    ax.axhline(trace.command_deg, color="green", linestyle="--", label="commanded angle")
    ax.plot(trace.t_ms, trace.actual_deg, color="tab:blue", label="actual angle")
    ax.axhline(trace.actual_deg[-1], color="tab:red", linestyle=":", label="where it settles")
    ax.annotate(
        f"settles {abs(trace.settled_gap_deg):.2f}° off the command",
        xy=(trace.t_ms[-1], trace.actual_deg[-1]),
        xytext=(0.4, 0.5), textcoords="axes fraction",
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_ylim(*_y_limits(trace, zoom))
    ax.set_title(trace.title)
    ax.set_xlabel("time since command (ms)")
    ax.grid(alpha=0.3)


def _y_limits(trace: Trace, zoom: bool) -> tuple[float, float]:
    """Axis range: command and settle point when zoomed, the full trajectory otherwise."""
    if zoom:
        lo, hi = sorted((trace.command_deg, float(trace.actual_deg[-1])))
        return lo - ZOOM_MARGIN_DEG, hi + ZOOM_MARGIN_DEG
    values = np.append(trace.actual_deg, trace.command_deg)
    return float(values.min()) - FULL_MARGIN_DEG, float(values.max()) + FULL_MARGIN_DEG


def build_figure(move: Trace, joint_name: str, out: Path) -> None:
    """The full move and a zoom on its end, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
    plot_trace(axes[0], move, zoom=False)
    plot_trace(axes[1], move, zoom=True)
    axes[1].set_title(f"{move.title} — zoomed")
    axes[0].set_ylabel(f"{joint_name} angle (deg)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Gravity vs. a spring-like position motor: the joint never settles on its command")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joint", default=DEFAULT_JOINT)
    parser.add_argument("--start-deg", type=float, default=DEFAULT_START_DEG)
    parser.add_argument("--target-deg", type=float, default=DEFAULT_TARGET_DEG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    """Simulate the move and write the figure."""
    args = parse_args()
    sim_cfg = SimConfig()
    joint = sim_cfg.joints.index(args.joint)
    backend = MujocoSo101Backend(sim_cfg)
    title = f"Settled at {args.start_deg:.0f}°, then commanded {args.target_deg:.0f}°"
    move = simulate(backend, joint, Case(title, np.deg2rad(args.start_deg), np.deg2rad(args.target_deg)))
    build_figure(move, args.joint, args.out)
    print(f"{move.title}: settled at {move.actual_deg[-1]:.2f} deg, gap {move.settled_gap_deg:+.2f} deg")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

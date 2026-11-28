"""Report figures for the hardware measurements, built from saved sag-check and pose-check data.

Writes to experiments/<date>_hardware_figures/:
  friction_bands.png   full friction band per joint for each procedure (small/large approaches, pose check)
  direction_errors.png error after arriving with vs against gravity, against gravity load, with the spring sag
  stiffness.png        MJCF kp vs the kp fitted to the large-approach session
  gain_sweep.png       largest miss per joint vs servo P gain, with c / P fits
  summary.json         every number plotted, plus the source folders
`--report-figures` copies the PNGs to reports/figures/ under the same names. Offline only.

Example (2026-10-31 data):
  python scripts/sim2real/analyze/hardware_figures.py \\
    --small-session data/sag_check/2026-10-31_rbe501_follower_p24_190722 \\
    --large-session data/sag_check/2026-10-31_rbe501_follower_p24_192311 \\
    --pose-check data/pose_check/2026-10-31_rbe501_follower_p24_184714 \\
    --gain-runs data/pose_check/2026-10-31_rbe501_follower_p16_184613 \\
      data/pose_check/2026-10-31_rbe501_follower_p24_184714 data/pose_check/2026-10-31_rbe501_follower_p32_184813 \\
    --report-figures
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rbe_501_so101.config import HardwareConfig, ReachTaskConfig, SagFitConfig, SimConfig, resolve_project_path
from rbe_501_so101.hardware.direction_errors import (
    DirectionalError,
    VisitError,
    directional_errors,
    opposite_arrival_bands,
    visit_errors,
)
from rbe_501_so101.hardware.figures import (
    BandSeries,
    plot_direction_errors,
    plot_friction_bands,
    plot_gain_sweep,
    plot_stiffness,
)
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.pose_check import PoseCheckRepository
from rbe_501_so101.hardware.sag import JointComplianceFit, SagRepository
from rbe_501_so101.hardware.sag_analysis import analyze_session
from rbe_501_so101.kinematics.gravity import GravityTorqueModel
from rbe_501_so101.provenance import begin_run, experiment_dir
from rbe_501_so101.reports import publish_figure

EXPERIMENT_SUFFIX = "hardware_figures"
RESULTS_FILE = "results.json"
LOADED_JOINTS = ("shoulder_lift", "elbow_flex")
FIGURES = ("friction_bands.png", "direction_errors.png", "stiffness.png", "gain_sweep.png")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--small-session", type=Path, required=True, help="sag session with small (3 deg) approaches")
    parser.add_argument("--large-session", type=Path, required=True, help="sag session with large (8 deg) approaches")
    parser.add_argument("--pose-check", type=Path, required=True, help="pose-check folder at the campaign gain")
    parser.add_argument("--gain-runs", type=Path, nargs="+", required=True, help="pose-check folders, one per P gain")
    parser.add_argument("--out", type=Path, default=None, help=f"default experiments/<date>_{EXPERIMENT_SUFFIX}")
    parser.add_argument("--report-figures", action="store_true", help="also copy the PNGs to reports/figures/")
    return parser.parse_args(argv)


class Context:
    """Model, joint map, and gravity model shared by every figure."""

    def __init__(self) -> None:
        self.sim = SimConfig()
        self.model = mujoco.MjModel.from_xml_path(str(self.sim.model_path))
        self.joint_map = JointMap.from_model(self.model, self.sim, HardwareConfig())
        self.gravity = GravityTorqueModel.from_config(self.sim)

    def mjcf_kp(self) -> dict[str, float]:
        """Actuator kp per arm joint as written in the MJCF."""
        ids = [self.model.actuator(j + self.sim.actuator_suffix).id for j in self.sim.arm_joints]
        return dict(zip(self.sim.arm_joints, map(float, self.model.actuator_gainprm[ids, 0]), strict=True))

    def directions(self, session_path: Path) -> tuple[DirectionalError, ...]:
        """With/against-gravity errors of the loaded joints in one sag session."""
        session = SagRepository().load_session(session_path)
        return directional_errors(session, self.joint_map, self.gravity, LOADED_JOINTS)

    def fits(self, session_path: Path) -> tuple[JointComplianceFit, ...]:
        """Stiffness fit per arm joint, as `analyze_sag.py` computes it."""
        session = SagRepository().load_session(session_path)
        analysis = analyze_session(
            session, self.joint_map, self.model, self.sim, SagFitConfig(), ReachTaskConfig().residual_max_rad
        )
        return tuple(j.fit for j in analysis.joints)

    def visits(self, folder: Path) -> tuple[VisitError, ...]:
        """Per-visit errors of one pose-check run, moving in from the rest pose."""
        results = PoseCheckRepository(folder / RESULTS_FILE).load()
        return visit_errors(results, HardwareConfig().rest_deg, self.gravity, self.sim.arm_joints)


def band_series(label: str, errors: tuple[DirectionalError, ...]) -> BandSeries:
    """Mean and range of the friction band per loaded joint."""
    by_joint = {j: [e.band_deg for e in errors if e.joint == j] for j in LOADED_JOINTS}
    return BandSeries(
        label,
        {j: float(np.mean(b)) for j, b in by_joint.items()},
        {j: (float(min(b)), float(max(b))) for j, b in by_joint.items()},
    )


def gain_of(folder: Path) -> int:
    """P gain read back from the servos in a pose-check run."""
    settings = PoseCheckRepository(folder / RESULTS_FILE).load_settings()
    return int(settings[0].p)


def largest_misses(ctx: Context, folders: list[Path]) -> tuple[list[int], dict[str, list[float]]]:
    """Gains (sorted) and the largest absolute miss per arm joint at each gain."""
    runs = sorted((gain_of(f), ctx.visits(f)) for f in folders)
    misses = {
        j: [max(abs(v.error_deg) for v in visits if v.joint == j) for _, visits in runs]
        for j in ctx.sim.arm_joints
    }
    return [g for g, _ in runs], misses


def make_figures(args: argparse.Namespace, out: Path) -> dict:
    """Render every figure into `out`; returns the plotted numbers."""
    ctx = Context()
    small, large = ctx.directions(resolve_project_path(args.small_session)), ctx.directions(
        resolve_project_path(args.large_session)
    )
    pose_bands = opposite_arrival_bands(ctx.visits(resolve_project_path(args.pose_check)))
    series = [band_series("3° approaches", small), band_series("8° approaches", large),
              BandSeries("pose check, 30° moves", {j: pose_bands[j] for j in LOADED_JOINTS if j in pose_bands})]
    plot_friction_bands(series, LOADED_JOINTS, out / FIGURES[0])
    fits = ctx.fits(resolve_project_path(args.large_session))
    kp = {f.joint: f.kp_nm_per_rad for f in fits if f.kp_nm_per_rad is not None}
    plot_direction_errors({"3° approaches": small, "8° approaches": large}, kp, LOADED_JOINTS, out / FIGURES[1])
    plot_stiffness(fits, ctx.mjcf_kp(), out / FIGURES[2])
    gains, misses = largest_misses(ctx, [resolve_project_path(f) for f in args.gain_runs])
    plot_gain_sweep(gains, misses, out / FIGURES[3])
    return {"bands_deg": {s.label: dict(s.mean_deg) for s in series}, "fitted_kp": kp,
            "mjcf_kp": ctx.mjcf_kp(), "gains": gains, "largest_miss_deg": misses,
            "sources": {k: str(v) for k, v in vars(args).items() if k != "report_figures"}}


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    out = resolve_project_path(args.out) if args.out else experiment_dir(EXPERIMENT_SUFFIX)
    manifest = begin_run(out)
    summary = make_figures(args, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if args.report_figures:
        for name in FIGURES:
            publish_figure(out / name, name)
    manifest.finalize(out)
    print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

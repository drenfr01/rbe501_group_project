"""Phase 0 analysis: sag, hysteresis, and fitted stiffness per joint from a saved sag session.

Maps the LeRobot-degree session into sim terms with the configured `JointMap`,
fits `sag = c * gravity_torque` per joint, and writes:
  experiments/<date>_sag_analysis/   report.json, README.md, real_sag.png, manifest.json
  data/calibration/actuator_calibration.json   (for `--calibration` / `SimConfig.calibration_path`)

Offline only; rerun any time from saved data.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco

from rbe_501_so101.config import (
    DEFAULT_CALIBRATION_FILE,
    HardwareConfig,
    ReachTaskConfig,
    SagFitConfig,
    SimConfig,
    resolve_project_path,
)
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.sag import SagRepository
from rbe_501_so101.hardware.sag_analysis import SagAnalysis, analyze_session, calibration_from_analysis, sag_readme
from rbe_501_so101.provenance import begin_run, experiment_dir
from rbe_501_so101.reports import publish_figure
from rbe_501_so101.rl.plots import plot_real_sag

PLOT_NAME = "real_sag.png"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path, help="session.json or the session folder")
    parser.add_argument("--out", type=Path, default=None, help="experiment folder (default experiments/<date>_sag_analysis)")
    parser.add_argument("--calibration-out", type=Path, default=Path(DEFAULT_CALIBRATION_FILE))
    parser.add_argument("--report-figures", action="store_true", help=f"also copy {PLOT_NAME} to reports/figures/")
    return parser.parse_args(argv)


def analyze(session_path: Path) -> SagAnalysis:
    """Load the session and fit it against the raw MJCF model."""
    sim_cfg = SimConfig()
    session = SagRepository().load_session(session_path)
    model = mujoco.MjModel.from_xml_path(str(sim_cfg.model_path))
    joint_map = JointMap.from_model(model, sim_cfg, HardwareConfig())
    fit_cfg = SagFitConfig()
    return analyze_session(session, joint_map, model, sim_cfg, fit_cfg, ReachTaskConfig().residual_max_rad)


def write_outputs(analysis: SagAnalysis, out: Path, calibration_out: Path, source: str) -> Path:
    """report.json, README, plot, and the calibration file; returns the plot path."""
    (out / "report.json").write_text(json.dumps(asdict(analysis), indent=2))
    calibration_from_analysis(analysis, source).save(calibration_out)
    readme = sag_readme(analysis) + f"\nSource session: `{source}`. Calibration written to `{calibration_out}`.\n"
    (out / "README.md").write_text(readme)
    plot_real_sag(analysis, out / PLOT_NAME)
    return out / PLOT_NAME


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    out = resolve_project_path(args.out) if args.out else experiment_dir(SagFitConfig().experiment_suffix)
    manifest = begin_run(out)
    session_path = resolve_project_path(args.session)
    analysis = analyze(session_path)
    plot = write_outputs(analysis, out, resolve_project_path(args.calibration_out), str(session_path))
    if args.report_figures:
        publish_figure(plot, PLOT_NAME)
    manifest.finalize(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

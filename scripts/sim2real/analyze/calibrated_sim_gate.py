"""Plan step 10: replay the real sag checks and pose check on the calibrated simulator and compare.

For each real session the simulator runs the same procedure (same goals,
approach offsets, waypoint, ramps, LeRobot relative clip, encoder rounding)
twice: with the calibration's stiffness and stiction (the gate), and with the
stiffness only (to show what stiction adds). Rows compare the mean error per
joint after arriving with vs against gravity. The gate passes if every row of
the stiction model is within `CalibratedSimGateConfig.tolerance_deg`.

Writes to experiments/<date>_calibrated_sim_gate/:
  README.md   PASS/FAIL and the table
  gate.png    real vs sim mean error per row
  report.json every row, the verdict, and the sources
`--report-figures` copies gate.png to reports/figures/calibrated_sim_gate.png. Offline only.

Example (2026-10-31 data):
  python scripts/sim2real/analyze/calibrated_sim_gate.py \\
    --sag-sessions data/sag_check/2026-10-31_rbe501_follower_p24_192311 \\
      data/sag_check/2026-10-31_rbe501_follower_p24_190722 \\
    --pose-check data/pose_check/2026-10-31_rbe501_follower_p24_184714 --report-figures
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import mujoco

from rbe_501_so101.config import (
    DEFAULT_CALIBRATION_FILE,
    CalibratedSimGateConfig,
    HardwareConfig,
    PoseCheckConfig,
    SagCheckConfig,
    ServoEffectsConfig,
    SimConfig,
    resolve_project_path,
)
from rbe_501_so101.hardware.direction_errors import VisitError, directional_errors, visit_errors
from rbe_501_so101.hardware.figures import plot_gate
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.pose_check import PoseCheckRepository, PoseCheckResult
from rbe_501_so101.hardware.sag import SagRepository, smallest_approach_deg
from rbe_501_so101.hardware.sim_gate import GateResult, GateRow, directional_rows, visit_rows
from rbe_501_so101.hardware.sim_replay import ReplaySetup, replay_pose_check, replay_sag_session
from rbe_501_so101.kinematics.gravity import GravityTorqueModel
from rbe_501_so101.provenance import begin_run, experiment_dir
from rbe_501_so101.reports import publish_figure

GATE_CFG = CalibratedSimGateConfig()
RESULTS_FILE = "results.json"
LOADED_JOINTS = ("shoulder_lift", "elbow_flex")
GATE_MODEL, PLAIN_MODEL = "sim: stiffness + stiction", "sim: stiffness only"
FIGURE = "gate.png"
REPORT_FIGURE = "calibrated_sim_gate.png"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sag-sessions", type=Path, nargs="+", required=True, help="real sag-check session folders")
    parser.add_argument("--pose-check", type=Path, required=True, help="real pose-check folder at the same gain")
    parser.add_argument("--calibration", type=Path, default=Path(DEFAULT_CALIBRATION_FILE))
    parser.add_argument("--tolerance-deg", type=float, default=GATE_CFG.tolerance_deg)
    parser.add_argument("--out", type=Path, default=None, help=f"default experiments/<date>_{GATE_CFG.experiment_suffix}")
    parser.add_argument("--report-figures", action="store_true", help=f"also copy {FIGURE} to reports/figures/")
    return parser.parse_args(argv)


class Gate:
    """Shared model, joint map, gravity model, and the two simulated arms."""

    def __init__(self, calibration: Path, tolerance_deg: float) -> None:
        self.sim = replace(SimConfig(), calibration_path=calibration)
        model = mujoco.MjModel.from_xml_path(str(self.sim.model_path))
        self.joint_map = JointMap.from_model(model, self.sim, HardwareConfig())
        self.gravity = GravityTorqueModel.from_config(self.sim)
        self.tolerance = tolerance_deg
        self.setups = {
            GATE_MODEL: ReplaySetup(self.sim, self.joint_map, ServoEffectsConfig(stiction=True)),
            PLAIN_MODEL: ReplaySetup(self.sim, self.joint_map, ServoEffectsConfig()),
        }

    def sag_rows(self, folder: Path) -> dict[str, list[GateRow]]:
        """Rows for one real sag session, per sim model."""
        real = SagRepository().load_session(folder)
        source = f"sag check, {smallest_approach_deg(real):g}° approaches"
        real_err = directional_errors(real, self.joint_map, self.gravity, LOADED_JOINTS)
        cfg, waypoint = SagCheckConfig(), SagCheckConfig().waypoint_sim_deg
        return {
            name: directional_rows(
                source,
                real_err,
                directional_errors(replay_sag_session(real, setup, cfg, waypoint), self.joint_map, self.gravity, LOADED_JOINTS),
                self.tolerance,
            )
            for name, setup in self.setups.items()
        }

    def pose_rows(self, folder: Path) -> dict[str, list[GateRow]]:
        """Rows for one real pose-check run, per sim model."""
        real = PoseCheckRepository(folder / RESULTS_FILE).load()
        names, cfg, poses = [r.name for r in real], PoseCheckConfig(), {p.name: p for p in PoseCheckConfig().poses}
        start = HardwareConfig().rest_deg
        real_v = self.loaded_visits(real, start)
        return {
            name: visit_rows(
                "pose check, 30° moves",
                real_v,
                self.loaded_visits(replay_pose_check(names, setup, cfg, poses), start),
                self.tolerance,
            )
            for name, setup in self.setups.items()
        }

    def loaded_visits(self, results: tuple[PoseCheckResult, ...], start: tuple[float, ...]) -> list[VisitError]:
        """Per-visit errors of the loaded joints."""
        visits = visit_errors(results, start, self.gravity, self.sim.arm_joints)
        return [v for v in visits if v.joint in LOADED_JOINTS]


def collect(args: argparse.Namespace) -> tuple[Gate, dict[str, list[GateRow]]]:
    """All rows, per sim model, from every sag session and the pose check."""
    gate = Gate(resolve_project_path(args.calibration), args.tolerance_deg)
    rows: dict[str, list[GateRow]] = {GATE_MODEL: [], PLAIN_MODEL: []}
    parts = [gate.sag_rows(resolve_project_path(f)) for f in args.sag_sessions]
    parts.append(gate.pose_rows(resolve_project_path(args.pose_check)))
    for part in parts:
        for name, part_rows in part.items():
            rows[name] += part_rows
    return gate, rows


def readme(result: GateResult, plain: GateResult) -> str:
    """PASS/FAIL and a table of every row (stiction model, with the stiffness-only error for contrast)."""
    lines = [
        "# Calibrated-sim gate",
        "",
        f"**{'PASS' if result.passed else 'FAIL'}**: every row within {result.tolerance_deg:g}° "
        f"({sum(r.passed for r in result.rows)}/{len(result.rows)} rows pass; "
        f"stiffness-only sim: {sum(r.passed for r in plain.rows)}/{len(plain.rows)}).",
        "",
        "Errors are sim degrees along gravity (positive = sagged). Columns: real mean, sim mean (stiffness + "
        "stiction), mean |real - sim|, the same for the stiffness-only sim.",
        "",
        "| source | joint | arrival | n | real | sim | diff | stiffness-only sim | its diff | pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r, p in zip(result.rows, plain.rows, strict=True):
        lines.append(
            f"| {r.source} | {r.joint} | {r.direction} | {r.n} | {r.real_mean_deg:+.2f} | {r.sim_mean_deg:+.2f} | "
            f"{r.mean_abs_diff_deg:.2f} | {p.sim_mean_deg:+.2f} | {p.mean_abs_diff_deg:.2f} | "
            f"{'yes' if r.passed else 'NO'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    out = resolve_project_path(args.out) if args.out else experiment_dir(GATE_CFG.experiment_suffix)
    manifest = begin_run(out)
    gate, rows = collect(args)
    result, plain = (GateResult(tuple(rows[m]), gate.tolerance) for m in (GATE_MODEL, PLAIN_MODEL))
    plot_gate(rows, gate.tolerance, out / FIGURE)
    (out / "README.md").write_text(readme(result, plain))
    report = {"passed": result.passed, "tolerance_deg": gate.tolerance, "rows": {m: [asdict(r) for r in rows[m]] for m in rows},
              "sources": {k: str(v) for k, v in vars(args).items() if k != "report_figures"}}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    if args.report_figures:
        publish_figure(out / FIGURE, REPORT_FIGURE)
    manifest.finalize(out)
    print((out / "README.md").read_text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

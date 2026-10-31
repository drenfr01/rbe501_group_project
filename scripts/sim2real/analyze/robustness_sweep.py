"""Phase 2: does the reach policy survive the expected sim-to-real gaps, in sim?

Evaluates the trained checkpoint (every seed) and the three classical
baselines (ik_hold, gravity_comp, integral) on the same targets under:
  - a stiffness sweep (every actuator kp scaled by the grid in RobustnessConfig);
  - encoder rounding with 0 and 1 control-step delay, alone and in the calibrated sim;
  - the reduced hardware command speed;
  - the calibrated sim (Phase 0 ActuatorCalibration), if `--calibration` is given.
Then applies the transfer-or-retrain rule to the calibrated (or, provisionally,
nominal) sim with rounding and the longest delay, and classifies the policy as
feedback- or feedforward-like from the stiffness sweep.

Writes experiments/<date>_reach_robustness/: report.json, README.md,
kp_sweep.png, saturation.png, manifest.json. Sim only; rerun at any time.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import mujoco

from rbe_501_so101.config import (
    EnvOptions,
    HardwareConfig,
    RobustnessConfig,
    SagCheckConfig,
    SimConfig,
    TrainConfig,
    resolve_project_path,
)
from rbe_501_so101.envs.calibration import ActuatorCalibration
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.sag import SagRepository
from rbe_501_so101.hardware.sag_analysis import JointSagComparison, compare_sag, simulate_session
from rbe_501_so101.provenance import begin_run, experiment_dir
from rbe_501_so101.reports import publish_figure
from rbe_501_so101.rl.checkpoints import CheckpointRepository
from rbe_501_so101.rl.robustness import (
    SweepCondition,
    SweepResult,
    classify_correction,
    evaluate_condition,
    measured_kp_scales,
    run_sensing_sweep,
    run_stiffness_sweep,
    sensing_options,
    transfer_decision,
    working_range,
)
from rbe_501_so101.rl.robustness_report import RobustnessReport, robustness_readme, write_plots

DEFAULT_MODEL_DIR = Path("checkpoints/reach_1m_residual_ik")
SPEED = "speed_scale"
CALIBRATED = "calibrated"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command line."""
    cfg = RobustnessConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=cfg.episodes_per_point, help="episodes per condition")
    parser.add_argument("--kp-scales", type=float, nargs="+", default=list(cfg.kp_scales))
    parser.add_argument("--calibration", type=Path, default=None, help="Phase 0 ActuatorCalibration JSON")
    parser.add_argument("--session", type=Path, default=None, help="Phase 0 sag session, for the calibrated-sim check")
    parser.add_argument("--speed-scale", type=float, default=HardwareConfig().speed_scale)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report-figures", action="store_true", help="also copy the plots to reports/figures/")
    return parser.parse_args(argv)


def calibrated_options(calibration: Path | None) -> EnvOptions:
    """Env options with the calibration file, or the nominal (placeholder) sim."""
    return EnvOptions(calibration_path=str(calibration)) if calibration else EnvOptions()


def decision_condition(base: EnvOptions, cfg: RobustnessConfig) -> SweepCondition:
    """Calibrated (or nominal) sim with encoder rounding and the longest configured delay."""
    delay = max(cfg.delay_steps)
    return SweepCondition("delay_steps", delay, sensing_options(delay, cfg, base))


def speed_results(
    repo: CheckpointRepository, seeds: tuple[int, ...], condition: SweepCondition, speed: float, cfg: RobustnessConfig
) -> list[SweepResult]:
    """The decision condition at full and at hardware command speed."""
    conditions = [replace(condition, parameter=SPEED, value=s, speed_scale=s) for s in sorted({1.0, speed})]
    return [r for c in conditions for r in evaluate_condition(repo, seeds, c, cfg)]


def sag_comparison(session: Path | None, calibration: Path | None) -> list[JointSagComparison]:
    """Calibrated sim run through the Phase 0 procedure at the real poses, against the real sag."""
    if session is None or calibration is None:
        return []
    sim_cfg = SimConfig(calibration_path=calibration)
    real = SagRepository().load_session(session)
    model = mujoco.MjModel.from_xml_path(str(sim_cfg.model_path))
    joint_map = JointMap.from_model(model, sim_cfg, HardwareConfig())
    sim = simulate_session(real, sim_cfg, joint_map, SagCheckConfig(), RobustnessConfig().quantization_rad)
    return list(compare_sag(real, sim, sim_cfg.arm_joints))


def kp_marks(calibration: Path | None) -> dict[str, float]:
    """Measured / MJCF stiffness per calibrated joint."""
    if calibration is None:
        return {}
    sim_cfg = SimConfig()
    model = mujoco.MjModel.from_xml_path(str(sim_cfg.model_path))
    nominal = model.actuator_gainprm[[model.actuator(j + sim_cfg.actuator_suffix).id for j in sim_cfg.arm_joints], 0]
    return measured_kp_scales(ActuatorCalibration.load(calibration), nominal, sim_cfg.arm_joints)


def build_report(args: argparse.Namespace, cfg: RobustnessConfig) -> RobustnessReport:
    """Run every sweep and draw the verdicts."""
    repo = CheckpointRepository(resolve_project_path(args.model_dir), TrainConfig())
    seeds, cal = tuple(args.seeds), resolve_project_path(args.calibration) if args.calibration else None
    base = calibrated_options(cal)
    stiffness = run_stiffness_sweep(repo, seeds, cfg)
    sensing = run_sensing_sweep(repo, seeds, cfg)
    sensing_cal = run_sensing_sweep(repo, seeds, cfg, base=base) if cal else []
    calibrated = evaluate_condition(repo, seeds, SweepCondition(CALIBRATED, 1.0, base), cfg) if cal else []
    decide = decision_condition(base, cfg)
    decision_results = [r for r in (sensing_cal or sensing) if r.value == decide.value]
    label = f"{'calibrated' if cal else 'nominal (placeholder kp)'} sim, encoder rounding, {decide.value:g}-step delay"
    return RobustnessReport(
        model_dir=str(args.model_dir), seeds=seeds, calibration_path=str(cal) if cal else None,
        stiffness=stiffness, sensing=sensing, sensing_calibrated=sensing_cal,
        speed=speed_results(repo, seeds, decide, args.speed_scale, cfg), calibrated=calibrated,
        decision_condition=label, decision=transfer_decision(decision_results, cfg),
        correction=classify_correction(stiffness), working_range=working_range(stiffness, cfg),
        measured_kp_scales=kp_marks(cal),
        sag_comparison=sag_comparison(resolve_project_path(args.session) if args.session else None, cal),
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    cfg = replace(RobustnessConfig(), episodes_per_point=args.episodes, kp_scales=tuple(args.kp_scales))
    out = resolve_project_path(args.out) if args.out else experiment_dir(cfg.experiment_suffix)
    manifest = begin_run(out)
    report = build_report(args, cfg)
    (out / "report.json").write_text(report.to_json())
    (out / "README.md").write_text(robustness_readme(report))
    for plot in write_plots(report, out):
        if args.report_figures:
            publish_figure(plot, plot.name)
    manifest.finalize(out)
    print(f"verdict: {'TRANSFER' if report.decision.transfer else 'RETRAIN'}; correction: {report.correction.kind} -> {out}")


if __name__ == "__main__":
    main()

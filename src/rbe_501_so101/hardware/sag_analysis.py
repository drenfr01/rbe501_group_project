"""Turn a Phase 0 session into sim terms: per-joint summary, calibration file, sim comparison, README."""

from dataclasses import dataclass

import mujoco
import numpy as np

from rbe_501_so101.config import (
    DynamicsRandomizationConfig,
    SagCheckConfig,
    SagFitConfig,
    SimConfig,
)
from rbe_501_so101.envs.calibration import ActuatorCalibration, JointActuatorCalibration
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.sag import (
    JointComplianceFit,
    PoseJointTable,
    RecordedPose,
    SagSession,
    fit_compliance,
    friction_torque,
    headroom_warnings,
    hysteresis_by_joint,
    sag_by_joint,
    usable_samples,
)
from rbe_501_so101.hardware.sag_procedure import SagCheckRunner
from rbe_501_so101.hardware.sim_bus import SimJointBus


@dataclass(frozen=True)
class JointSagSummary:
    """One arm joint's Phase 0 numbers next to the MJCF placeholder stiffness."""

    joint: str
    mean_sag_deg: float
    max_abs_sag_deg: float
    mean_abs_hysteresis_deg: float
    mjcf_kp: float
    fit: JointComplianceFit
    friction_nm: float | None


@dataclass(frozen=True)
class SagAnalysis:
    """Everything the Phase 0 experiment README reports."""

    robot_id: str
    n_poses: int
    joints: tuple[JointSagSummary, ...]
    headroom_warnings: tuple[str, ...]
    residual_max_rad: float
    warning_fraction: float
    min_hysteresis_deg: float
    n_excluded_clamped: int = 0


@dataclass(frozen=True)
class JointSagComparison:
    """Mean sag of one joint on the real arm and in the (calibrated) sim at the same poses."""

    joint: str
    real_mean_sag_deg: float
    sim_mean_sag_deg: float


def analyze_session(
    session: SagSession,
    joint_map: JointMap,
    model: mujoco.MjModel,
    sim_cfg: SimConfig,
    fit_cfg: SagFitConfig,
    residual_max_rad: float,
) -> SagAnalysis:
    """Sag, hysteresis, and stiffness fit per arm joint, plus the headroom early warning.

    Samples flagged as clamped (LeRobot clipped a goal) are excluded first.
    """
    usable = usable_samples(session)
    sag = sag_by_joint(usable)
    joints = _joint_summaries(usable, sag, joint_map, model, sim_cfg, fit_cfg)
    warnings = headroom_warnings(sag, joint_map, residual_max_rad, fit_cfg.early_warning_fraction)
    return SagAnalysis(
        session.robot_id, len(sag.pose_ids), joints, warnings, residual_max_rad, fit_cfg.early_warning_fraction,
        fit_cfg.min_hysteresis_deg, len(session.samples) - len(usable.samples),
    )


def _joint_summaries(
    session: SagSession, sag: PoseJointTable, joint_map: JointMap, model: mujoco.MjModel, sim_cfg: SimConfig,
    fit_cfg: SagFitConfig,
) -> tuple[JointSagSummary, ...]:
    """Per-arm-joint sag, hysteresis, and stiffness fit against the MJCF kp."""
    hysteresis = hysteresis_by_joint(session)
    fits = fit_compliance(session, joint_map, model, sim_cfg, fit_cfg)
    kp_mjcf = model.actuator_gainprm[[model.actuator(j + sim_cfg.actuator_suffix).id for j in sim_cfg.arm_joints], 0]
    return tuple(
        _joint_summary(i, fit, sag.values_deg, hysteresis.values_deg, float(kp_mjcf[i]), joint_map)
        for i, fit in enumerate(fits)
    )


def calibration_from_analysis(analysis: SagAnalysis, source: str) -> ActuatorCalibration:
    """Measured stiffness for identifiable joints; friction only where hysteresis exceeds one count."""
    entries = []
    for joint in analysis.joints:
        if not joint.fit.identifiable or joint.fit.kp_nm_per_rad is None:
            continue
        friction = joint.friction_nm if joint.mean_abs_hysteresis_deg >= analysis.min_hysteresis_deg else None
        entries.append(JointActuatorCalibration(joint.joint, joint.fit.kp_nm_per_rad, friction))
    return ActuatorCalibration(tuple(entries), source=source)


def simulate_session(
    real: SagSession, sim_cfg: SimConfig, joint_map: JointMap, cfg: SagCheckConfig, quantization_rad: float
) -> SagSession:
    """Run the Phase 0 procedure in sim (with `sim_cfg.calibration_path`, if set) at the real poses."""
    no_randomization = DynamicsRandomizationConfig(enabled=False)
    bus = SimJointBus(MujocoSo101Backend(sim_cfg, no_randomization), joint_map, sim_cfg, quantization_rad)
    goals = {s.pose_id: s.goal_deg for s in usable_samples(real).samples}
    poses = tuple(RecordedPose(pid, goal) for pid, goal in goals.items())
    try:
        return SagCheckRunner(bus, bus, cfg, real.joints).session("sim", poses)
    finally:
        bus.close()


def compare_sag(real: SagSession, sim: SagSession, joints: tuple[str, ...]) -> tuple[JointSagComparison, ...]:
    """Mean sag per joint, real against sim, over the same poses (clamped real samples excluded)."""
    real_mean = sag_by_joint(usable_samples(real)).values_deg.mean(axis=0)
    sim_mean = sag_by_joint(sim).values_deg.mean(axis=0)
    index = {name: i for i, name in enumerate(real.joints)}
    return tuple(JointSagComparison(j, float(real_mean[index[j]]), float(sim_mean[index[j]])) for j in joints)


def sag_readme(analysis: SagAnalysis) -> str:
    """Markdown README for the Phase 0 experiment folder."""
    lines = [
        f"# Phase 0 sag check: {analysis.robot_id}",
        "",
        f"{analysis.n_poses} poses, each approached from above and from below. Sag is commanded minus",
        "measured (LeRobot degrees); hysteresis is half of (from above minus from below).",
        *_exclusion_lines(analysis),
        "",
        "| joint | mean sag (deg) | max abs sag (deg) | mean abs hysteresis (deg) | MJCF kp | fitted kp (95% CI) | friction (N*m) |",
        "|---|---|---|---|---|---|---|",
        *(_joint_row(j) for j in analysis.joints),
        "",
        *_warning_lines(analysis),
    ]
    return "\n".join(lines) + "\n"


def _joint_summary(
    index: int, fit: JointComplianceFit, sag: np.ndarray, hysteresis: np.ndarray, mjcf_kp: float, joint_map: JointMap
) -> JointSagSummary:
    """Summary row for arm joint `index`."""
    hyst_deg = float(np.mean(np.abs(hysteresis[:, index]))) if len(hysteresis) else 0.0
    friction = None
    if fit.kp_nm_per_rad is not None:
        friction = float(friction_torque(np.deg2rad(hyst_deg), np.array(fit.kp_nm_per_rad)))
    return JointSagSummary(
        joint=fit.joint,
        mean_sag_deg=float(np.mean(sag[:, index])),
        max_abs_sag_deg=float(np.max(np.abs(sag[:, index]))),
        mean_abs_hysteresis_deg=hyst_deg,
        mjcf_kp=mjcf_kp,
        fit=fit,
        friction_nm=friction,
    )


def _exclusion_lines(analysis: SagAnalysis) -> list[str]:
    """Note on samples dropped because LeRobot clipped a goal."""
    if not analysis.n_excluded_clamped:
        return []
    return [
        "",
        f"**{analysis.n_excluded_clamped} samples excluded:** LeRobot's `max_relative_target` clipped a goal sent for",
        "them, so the arm was not commanded to the recorded pose.",
    ]


def _joint_row(j: JointSagSummary) -> str:
    """One README table row."""
    if j.fit.identifiable and j.fit.kp_nm_per_rad is not None:
        kp = f"{j.fit.kp_nm_per_rad:.2f} ({j.fit.kp_ci_low:.2f} to {j.fit.kp_ci_high:.2f})"
    else:
        kp = f"not identifiable (torque span {j.fit.torque_span_nm:.3f} N*m)"
    friction = "-" if j.friction_nm is None else f"{j.friction_nm:.3f}"
    return (
        f"| {j.joint} | {j.mean_sag_deg:.2f} | {j.max_abs_sag_deg:.2f} | {j.mean_abs_hysteresis_deg:.2f} "
        f"| {j.mjcf_kp:.1f} | {kp} | {friction} |"
    )


def _warning_lines(analysis: SagAnalysis) -> list[str]:
    """The early-warning statement against the policy's correction cap."""
    limit_deg = np.rad2deg(analysis.warning_fraction * analysis.residual_max_rad)
    header = (
        f"**Early warning** (sag above {analysis.warning_fraction:.0%} of the policy's "
        f"±{analysis.residual_max_rad:g} rad correction cap, i.e. {limit_deg:.2f} deg):"
    )
    if not analysis.headroom_warnings:
        return [header, "no joint exceeds it; the existing policy should have correction headroom."]
    joints = ", ".join(analysis.headroom_warnings)
    return [header, f"{joints} exceed it; the policy will likely run out of correction room. Plan on retraining."]

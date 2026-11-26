"""Turn a Phase 0 session into sim terms: per-joint summary, calibration file, sim comparison, README."""

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from rbe_501_so101.config import SagCheckConfig, SagFitConfig, SimConfig
from rbe_501_so101.envs.calibration import ActuatorCalibration, JointActuatorCalibration
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.sag import (
    JointComplianceFit,
    PoseJointTable,
    SagSession,
    StictionFit,
    fit_compliance,
    fit_stiction,
    friction_torque,
    headroom_warnings,
    hysteresis_by_joint,
    one_sided_sag_by_joint,
    sag_by_joint,
    usable_samples,
)
from rbe_501_so101.hardware.sim_replay import ReplaySetup, replay_sag_session


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
    stiction: StictionFit | None = None


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
    assume_unidentified_kp: bool = False


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
    """Sag, hysteresis, stiffness and stiction fits per arm joint, plus the headroom early warning.

    Samples flagged as clamped (LeRobot clipped a goal) are excluded first.
    The warning uses the worst single arrival, since that is what a policy
    correction has to cover.
    """
    usable = usable_samples(session)
    sag = sag_by_joint(usable)
    joints = _joint_summaries(usable, sag, joint_map, model, sim_cfg, fit_cfg)
    one_sided = one_sided_sag_by_joint(usable)
    warnings = headroom_warnings(one_sided, joint_map, residual_max_rad, fit_cfg.early_warning_fraction)
    return SagAnalysis(
        session.robot_id, len(sag.pose_ids), joints, warnings, residual_max_rad, fit_cfg.early_warning_fraction,
        fit_cfg.min_hysteresis_deg, len(session.samples) - len(usable.samples), fit_cfg.assume_unidentified_kp,
    )


def _joint_summaries(
    session: SagSession, sag: PoseJointTable, joint_map: JointMap, model: mujoco.MjModel, sim_cfg: SimConfig,
    fit_cfg: SagFitConfig,
) -> tuple[JointSagSummary, ...]:
    """Per-arm-joint sag, hysteresis, and stiffness fit against the MJCF kp."""
    hysteresis = hysteresis_by_joint(session)
    fits = fit_compliance(session, joint_map, model, sim_cfg, fit_cfg)
    stiction = {s.joint: s for s in fit_stiction(session, joint_map, model, sim_cfg, fit_cfg)}
    kp_mjcf = model.actuator_gainprm[[model.actuator(j + sim_cfg.actuator_suffix).id for j in sim_cfg.arm_joints], 0]
    return tuple(
        replace(
            _joint_summary(i, fit, sag.values_deg, hysteresis.values_deg, float(kp_mjcf[i]), joint_map),
            stiction=stiction.get(fit.joint),
        )
        for i, fit in enumerate(fits)
    )


def calibration_from_analysis(analysis: SagAnalysis, source: str) -> ActuatorCalibration:
    """Measured stiffness (or the assumed mean where unmeasured) and the friction model per joint.

    A stiction fit replaces MuJoCo `frictionloss` (which creeps, D-03). Without
    one, friction falls back to `kp * hysteresis` where hysteresis exceeds one count.
    """
    fitted = [j.fit.kp_nm_per_rad for j in analysis.joints if _has_kp(j)]
    assumed = float(np.mean(fitted)) if fitted and analysis.assume_unidentified_kp else None
    entries = (_calibration_entry(j, assumed, analysis.min_hysteresis_deg) for j in analysis.joints)
    return ActuatorCalibration(tuple(e for e in entries if e is not None), source=source)


def _has_kp(joint: JointSagSummary) -> bool:
    """True if the joint's stiffness was identified."""
    return joint.fit.identifiable and joint.fit.kp_nm_per_rad is not None


def _calibration_entry(
    joint: JointSagSummary, assumed_kp: float | None, min_hysteresis_deg: float
) -> JointActuatorCalibration | None:
    """One joint's calibration entry, or None when nothing about it was measured."""
    kp, kp_assumed = (joint.fit.kp_nm_per_rad, False) if _has_kp(joint) else (assumed_kp, assumed_kp is not None)
    if joint.stiction is not None:
        return JointActuatorCalibration(
            joint.joint, kp, None, kp_assumed, joint.stiction.half_band_rad, joint.stiction.load_slope_rad_per_nm
        )
    if kp is None:
        return None
    friction = joint.friction_nm if joint.mean_abs_hysteresis_deg >= min_hysteresis_deg and not kp_assumed else None
    return JointActuatorCalibration(joint.joint, kp, friction, kp_assumed)


def simulate_session(
    real: SagSession, sim_cfg: SimConfig, joint_map: JointMap, cfg: SagCheckConfig, quantization_rad: float
) -> SagSession:
    """Run the Phase 0 procedure in sim (with `sim_cfg.calibration_path`, if set) at the real poses.

    Plain MuJoCo arm from the sim home pose: no servo effects, clip, or waypoint
    (see `sim_replay.replay_sag_session` for the full replay).
    """
    setup = ReplaySetup(sim_cfg, joint_map, quantization_rad=quantization_rad, max_relative_deg=None, start_sim_deg=None)
    return replay_sag_session(real, setup, cfg, waypoint_sim_deg=None)


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
        "| joint | mean sag (deg) | max abs sag (deg) | mean abs hysteresis (deg) | MJCF kp | fitted kp (95% CI) "
        "| friction (N*m) | stiction half-band (deg) |",
        "|---|---|---|---|---|---|---|---|",
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
        f"| {j.mjcf_kp:.1f} | {kp} | {friction} | {_stiction_text(j.stiction)} |"
    )


def _stiction_text(fit: StictionFit | None) -> str:
    """`b0 + slope * |tau|` in degrees, or a dash when the session could not measure it."""
    if fit is None:
        return "- (approaches too small)"
    b0 = np.rad2deg(fit.half_band_rad)
    if fit.load_slope_rad_per_nm == 0.0:
        return f"{b0:.2f}"
    return f"{b0:.2f} + {np.rad2deg(fit.load_slope_rad_per_nm):.2f}·|τ|"


def _warning_lines(analysis: SagAnalysis) -> list[str]:
    """The early-warning statement against the policy's correction cap."""
    limit_deg = np.rad2deg(analysis.warning_fraction * analysis.residual_max_rad)
    header = (
        f"**Early warning** (worst single-approach sag above {analysis.warning_fraction:.0%} of the policy's "
        f"±{analysis.residual_max_rad:g} rad correction cap, i.e. {limit_deg:.2f} deg):"
    )
    if not analysis.headroom_warnings:
        return [header, "no joint exceeds it; the existing policy should have correction headroom."]
    joints = ", ".join(analysis.headroom_warnings)
    return [header, f"{joints} exceed it; the policy will likely run out of correction room. Plan on retraining."]

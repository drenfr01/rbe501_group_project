"""Phase 0 sag check: session data, its repository, and the sag/hysteresis/stiffness fits.

Sag is commanded minus measured. Hysteresis is half of (measured arriving
from above minus measured arriving from below). Both are recorded in LeRobot
degrees and only converted to sim radians (with the Phase 1 `JointMap`) for fits.
"""

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from rbe_501_so101.config import SagCheckConfig, SagFitConfig, SimConfig
from rbe_501_so101.envs.model_indices import ModelIndices
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.types import MotorSettings, MotorStatus
from rbe_501_so101.kinematics.gravity import GravityTorqueModel
from rbe_501_so101.stats import student_t_ppf

Approach = Literal["above", "below"]
APPROACHES: tuple[Approach, Approach] = ("above", "below")


@dataclass(frozen=True)
class RecordedPose:
    """One hand-jogged sag-check pose, in LeRobot units and session joint order."""

    pose_id: int
    positions_deg: tuple[float, ...]


@dataclass(frozen=True)
class SagSample:
    """One settled arrival at a pose, in LeRobot degrees."""

    pose_id: int
    approach: Approach
    goal_deg: tuple[float, ...]
    measured_deg: tuple[float, ...]
    measured_std_deg: tuple[float, ...]
    status: tuple[MotorStatus, ...]


@dataclass(frozen=True)
class SagSession:
    """One Phase 0 session: servo settings plus every sample."""

    robot_id: str
    joints: tuple[str, ...]
    settings: tuple[MotorSettings, ...]
    samples: tuple[SagSample, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready nested dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SagSession":
        """Inverse of `to_dict`."""
        return cls(
            robot_id=data["robot_id"],
            joints=tuple(data["joints"]),
            settings=tuple(MotorSettings(**s) for s in data["settings"]),
            samples=tuple(_sample_from_dict(s) for s in data["samples"]),
        )


@dataclass(frozen=True)
class PoseJointTable:
    """A per-pose, per-joint quantity in LeRobot degrees (rows follow `pose_ids`)."""

    pose_ids: tuple[int, ...]
    joints: tuple[str, ...]
    values_deg: np.ndarray


@dataclass(frozen=True)
class JointComplianceFit:
    """Least-squares fit of `sag = c * gravity_torque` (through the origin) for one joint.

    `n_samples` counts poses, since each pose contributes one averaged point.

    Stiffness `kp = 1/c`; its interval inverts the interval on `c` (upper bound
    is infinite when the lower bound on `c` is not positive).
    """

    joint: str
    identifiable: bool
    n_samples: int
    torque_span_nm: float
    compliance_rad_per_nm: float | None = None
    compliance_ci: tuple[float, float] | None = None
    kp_nm_per_rad: float | None = None
    kp_ci_low: float | None = None
    kp_ci_high: float | None = None


class SagRepository:
    """Owns `data/sag_check/`: the recorded pose list and one folder per session."""

    def __init__(self, cfg: SagCheckConfig = SagCheckConfig()) -> None:
        self._cfg = cfg

    def session_dir(self, robot_id: str, day: date | None = None) -> Path:
        """`<sessions_dir>/<YYYY-MM-DD>_<robot_id>/`."""
        return self._cfg.sessions_dir / f"{(day or date.today()).isoformat()}_{robot_id}"

    def save_session(self, session: SagSession, folder: Path) -> Path:
        """Write `session.json` into `folder` and return its path."""
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / self._cfg.session_filename
        path.write_text(json.dumps(session.to_dict(), indent=2))
        return path

    def load_session(self, path: Path) -> SagSession:
        """Read a session from `session.json` (or the folder containing it)."""
        path = path / self._cfg.session_filename if path.is_dir() else path
        return SagSession.from_dict(json.loads(path.read_text()))

    def save_poses(self, joints: tuple[str, ...], poses: tuple[RecordedPose, ...]) -> Path:
        """Write the recorded pose list to `poses_path`."""
        self._cfg.poses_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"joints": list(joints), "poses": [asdict(p) for p in poses]}
        self._cfg.poses_path.write_text(json.dumps(payload, indent=2))
        return self._cfg.poses_path

    def load_poses(self) -> tuple[tuple[str, ...], tuple[RecordedPose, ...]]:
        """Joint names and poses from `poses_path`."""
        data = json.loads(self._cfg.poses_path.read_text())
        poses = tuple(RecordedPose(p["pose_id"], tuple(p["positions_deg"])) for p in data["poses"])
        return tuple(data["joints"]), poses


def sag_by_joint(session: SagSession) -> PoseJointTable:
    """Mean sag per pose and joint, averaged over both approach directions."""
    pose_ids = _pose_ids(session)
    rows = [np.mean([_sag(s) for s in session.samples if s.pose_id == pid], axis=0) for pid in pose_ids]
    return PoseJointTable(pose_ids, session.joints, np.array(rows))


def hysteresis_by_joint(session: SagSession) -> PoseJointTable:
    """Half of (measured from above minus measured from below), for poses with both."""
    pose_ids = tuple(pid for pid in _pose_ids(session) if _has_both(session, pid))
    rows = [(_measured(session, pid, "above") - _measured(session, pid, "below")) / 2 for pid in pose_ids]
    return PoseJointTable(pose_ids, session.joints, np.array(rows).reshape(len(pose_ids), len(session.joints)))


def headroom_warnings(
    sag: PoseJointTable, joint_map: JointMap, residual_max_rad: float, fraction: float
) -> tuple[str, ...]:
    """Arm joints whose largest sag exceeds `fraction` of the policy's correction cap."""
    n_arm = len(joint_map.joints) - 1
    sag_rad = np.abs([joint_map.arm_delta_to_sim(row[:n_arm]) for row in sag.values_deg])
    worst = sag_rad.max(axis=0)
    return tuple(name for name, w in zip(joint_map.joints[:n_arm], worst, strict=True) if w > fraction * residual_max_rad)


def friction_torque(hysteresis_rad: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Coulomb friction estimate `kp * |hysteresis|`: the start value for `dof_frictionloss`."""
    return np.asarray(kp) * np.abs(np.asarray(hysteresis_rad))


def fit_compliance(
    session: SagSession,
    joint_map: JointMap,
    model: mujoco.MjModel,
    sim_cfg: SimConfig = SimConfig(),
    cfg: SagFitConfig = SagFitConfig(),
) -> tuple[JointComplianceFit, ...]:
    """Fit compliance per arm joint from sag against the model's gravity torque at the measured pose."""
    gravity = GravityTorqueModel(model, ModelIndices.from_model(model, sim_cfg))
    torques, sags = _torques_and_sags(session, joint_map, gravity)
    return tuple(_fit_joint(name, torques[:, j], sags[:, j], cfg) for j, name in enumerate(sim_cfg.arm_joints))


def _torques_and_sags(
    session: SagSession, joint_map: JointMap, gravity: GravityTorqueModel
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pose arm gravity torque (N*m) and direction-averaged sag (sim radians).

    The two approaches to one pose are not independent (they often settle on
    the same encoder count), so each pose contributes one point to the fit.
    """
    n_arm = len(joint_map.joints) - 1
    torques, sags = [], []
    for pose_id in _pose_ids(session):
        samples = [s for s in session.samples if s.pose_id == pose_id]
        measured = np.mean([s.measured_deg for s in samples], axis=0)
        reading = {pos_key(j): float(v) for j, v in zip(session.joints, measured, strict=True)}
        torques.append(gravity.torque(joint_map.to_sim(reading)))
        sags.append(joint_map.arm_delta_to_sim(np.mean([_sag(s) for s in samples], axis=0)[:n_arm]))
    return np.array(torques), np.array(sags)


def _fit_joint(joint: str, torque: np.ndarray, sag: np.ndarray, cfg: SagFitConfig) -> JointComplianceFit:
    """Origin-constrained least squares with a Student-t interval."""
    span = float(np.ptp(torque)) if len(torque) else 0.0
    n = len(torque)
    if span < cfg.min_torque_span_nm or n < cfg.min_samples:
        return JointComplianceFit(joint, identifiable=False, n_samples=n, torque_span_nm=span)
    c = float(torque @ sag / (torque @ torque))
    dof = n - 1
    stderr = math.sqrt(float(np.sum((sag - c * torque) ** 2)) / dof / float(torque @ torque))
    half = student_t_ppf(0.5 + cfg.confidence / 2, dof) * stderr
    return JointComplianceFit(
        joint,
        identifiable=True,
        n_samples=n,
        torque_span_nm=span,
        compliance_rad_per_nm=c,
        compliance_ci=(c - half, c + half),
        kp_nm_per_rad=1.0 / c if c > 0 else None,
        kp_ci_low=1.0 / (c + half) if c + half > 0 else None,
        kp_ci_high=1.0 / (c - half) if c - half > 0 else math.inf,
    )


def _sag(sample: SagSample) -> np.ndarray:
    """Commanded minus measured, in LeRobot degrees."""
    return np.asarray(sample.goal_deg) - np.asarray(sample.measured_deg)


def _pose_ids(session: SagSession) -> tuple[int, ...]:
    """Pose ids in first-seen order."""
    return tuple(dict.fromkeys(s.pose_id for s in session.samples))


def _has_both(session: SagSession, pose_id: int) -> bool:
    """True if the pose was reached from above and from below."""
    return {s.approach for s in session.samples if s.pose_id == pose_id} == set(APPROACHES)


def _measured(session: SagSession, pose_id: int, approach: Approach) -> np.ndarray:
    """Mean measured position over the matching samples."""
    rows = [s.measured_deg for s in session.samples if s.pose_id == pose_id and s.approach == approach]
    return np.mean(rows, axis=0)


def _sample_from_dict(data: dict[str, Any]) -> SagSample:
    """Rebuild one sample from JSON."""
    return SagSample(
        pose_id=data["pose_id"],
        approach=data["approach"],
        goal_deg=tuple(data["goal_deg"]),
        measured_deg=tuple(data["measured_deg"]),
        measured_std_deg=tuple(data["measured_std_deg"]),
        status=tuple(MotorStatus(**m) for m in data["status"]),
    )

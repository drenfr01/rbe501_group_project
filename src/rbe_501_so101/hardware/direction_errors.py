"""Servo error split by arrival direction: arriving with gravity or against it.

Errors are measured minus goal in sim degrees, signed positive in the direction
gravity pulls the joint. A pure spring sags the same amount from both sides;
friction makes a with-gravity arrival land short of the spring sag and an
against-gravity arrival land past it. Their difference is the friction band.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rbe_501_so101.config import SagCheckConfig
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.pose_check import PoseCheckResult
from rbe_501_so101.hardware.sag import APPROACHES, SagSample, SagSession
from rbe_501_so101.kinematics.gravity import GravityTorqueModel

LOAD_FLOOR_NM = SagCheckConfig().gravity_torque_floor_nm


@dataclass(frozen=True)
class DirectionalError:
    """One loaded joint at one sag-check pose, arrived at from both sides."""

    pose_id: int
    joint: str
    gravity_load_nm: float
    with_gravity_deg: float
    against_gravity_deg: float

    @property
    def band_deg(self) -> float:
        """Full friction band: against-gravity minus with-gravity error."""
        return self.against_gravity_deg - self.with_gravity_deg

    @property
    def midpoint_deg(self) -> float:
        """Direction-averaged error: the spring (gravity) sag if friction is symmetric."""
        return (self.against_gravity_deg + self.with_gravity_deg) / 2


@dataclass(frozen=True)
class VisitError:
    """One joint at one pose-check visit.

    `arrived_with_gravity` is None when the joint carries no gravity load or did
    not move on the way in; `error_deg` is then the raw sim error.
    """

    visit: int
    name: str
    joint: str
    gravity_load_nm: float
    error_deg: float
    arrived_with_gravity: bool | None


def pull_sign(holding_torque_nm: np.ndarray) -> np.ndarray:
    """Sim direction gravity pulls each joint: opposite the torque needed to hold it."""
    return -np.sign(np.asarray(holding_torque_nm, dtype=float))


def directional_errors(
    session: SagSession, joint_map: JointMap, gravity: GravityTorqueModel, joints: Sequence[str]
) -> tuple[DirectionalError, ...]:
    """One entry per pose (reached from both sides) and loaded joint in `joints`."""
    out: list[DirectionalError] = []
    for pose_id in dict.fromkeys(s.pose_id for s in session.samples):
        pair = [s for s in session.samples if s.pose_id == pose_id]
        if {s.approach for s in pair} == set(APPROACHES):
            out.extend(_pose_errors(pose_id, pair, session.joints, joint_map, gravity, joints))
    return tuple(out)


def _pose_errors(
    pose_id: int,
    pair: list[SagSample],
    session_joints: tuple[str, ...],
    joint_map: JointMap,
    gravity: GravityTorqueModel,
    joints: Sequence[str],
) -> list[DirectionalError]:
    """Entries for the loaded joints of one pose."""
    goal = {pos_key(j): v for j, v in zip(session_joints, pair[0].goal_deg, strict=True)}
    tau = gravity.torque(joint_map.to_sim(goal))
    arm = joint_map.joints[:-1]
    out = []
    for joint in joints:
        i = arm.index(joint)
        if abs(tau[i]) > LOAD_FLOOR_NM:
            out.append(_entry(pose_id, joint, i, float(tau[i]), pair, joint_map))
    return out


def _entry(
    pose_id: int, joint: str, i: int, tau: float, pair: list[SagSample], joint_map: JointMap
) -> DirectionalError:
    """Assign the two samples of one joint to with/against gravity."""
    pull = float(pull_sign(tau))
    errors = {_arrived_with_gravity(s, i, pull, joint_map): pull * _sim_error_deg(s, i, joint_map) for s in pair}
    return DirectionalError(pose_id, joint, abs(tau), errors[True], errors[False])


def _sim_error_deg(sample: SagSample, i: int, joint_map: JointMap) -> float:
    """Measured minus goal of arm joint `i`, in sim degrees."""
    n_arm = len(joint_map.joints) - 1
    delta = np.asarray(sample.measured_deg[:n_arm]) - np.asarray(sample.goal_deg[:n_arm])
    return float(np.rad2deg(joint_map.arm_delta_to_sim(delta))[i])


def _arrived_with_gravity(sample: SagSample, i: int, pull: float, joint_map: JointMap) -> bool:
    """True if the sample started up-gravity of its goal, so its last move followed gravity.

    Sessions without recorded offsets used uniform offsets: "above" at +LeRobot.
    """
    n_arm = len(joint_map.joints) - 1
    if sample.approach_offset_deg is None:
        offset = np.full(n_arm, 1.0 if sample.approach == "above" else -1.0)
    else:
        offset = np.asarray(sample.approach_offset_deg[:n_arm])
    return bool(joint_map.arm_delta_to_sim(offset)[i] * pull < 0)


def visit_errors(
    results: Sequence[PoseCheckResult],
    start_sim_deg: Sequence[float],
    gravity: GravityTorqueModel,
    joints: Sequence[str],
) -> tuple[VisitError, ...]:
    """Per-visit, per-joint errors; the move direction comes from the previous goal (or the start)."""
    out: list[VisitError] = []
    previous = np.asarray(start_sim_deg[: len(joints)], dtype=float)
    for visit, result in enumerate(results):
        goal = np.asarray(result.sim_goal_deg[: len(joints)], dtype=float)
        error = np.asarray(result.measured_sim_deg[: len(joints)]) - goal
        tau = gravity.torque(np.append(np.deg2rad(goal), 0.0))
        out.extend(
            _visit_entry(visit, result.name, j, float(tau[i]), goal[i] - previous[i], error[i])
            for i, j in enumerate(joints)
        )
        previous = goal
    return tuple(out)


def _visit_entry(visit: int, name: str, joint: str, tau: float, move_deg: float, error_deg: float) -> VisitError:
    """Sign the error along gravity and classify the arrival, when the joint is loaded and moved."""
    if abs(tau) <= LOAD_FLOOR_NM or move_deg == 0.0:
        return VisitError(visit, name, joint, abs(tau), float(error_deg), None)
    pull = float(pull_sign(tau))
    return VisitError(visit, name, joint, abs(tau), float(pull * error_deg), bool(move_deg * pull > 0))


def opposite_arrival_bands(visits: Sequence[VisitError]) -> dict[str, float]:
    """Per joint, the largest against-minus-with error at a pose visited from both sides."""
    bands: dict[str, float] = {}
    for joint, name in dict.fromkeys((v.joint, v.name) for v in visits):
        same = [v for v in visits if v.joint == joint and v.name == name]
        against = [v.error_deg for v in same if v.arrived_with_gravity is False]
        along = [v.error_deg for v in same if v.arrived_with_gravity is True]
        if against and along:
            bands[joint] = max(bands.get(joint, -np.inf), against[0] - along[0])
    return bands

"""Phase 0 procedure: approach each pose from above and below, settle, and average readings."""

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol

import numpy as np

from rbe_501_so101.config import SagCheckConfig
from rbe_501_so101.hardware.bus import JointBus
from rbe_501_so101.hardware.clock import Clock
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.sag import APPROACHES, Approach, RecordedPose, SagSample, SagSession
from rbe_501_so101.kinematics.gravity import GravityTorqueModel

APPROACH_SIGN: dict[Approach, float] = {"above": 1.0, "below": -1.0}


def with_approach_offset(cfg: SagCheckConfig, offset_deg: float) -> SagCheckConfig:
    """`cfg` with a new approach offset; refuses offsets outside `approach_offset_range_deg`."""
    low, high = cfg.approach_offset_range_deg
    if not low <= offset_deg <= high:
        raise ValueError(f"approach offset {offset_deg} deg outside [{low}, {high}] deg")
    return replace(cfg, approach_offset_deg=offset_deg)


def lerobot_waypoint(joint_map: JointMap, waypoint_sim_deg: tuple[float, ...]) -> dict[str, float]:
    """The raised waypoint (sim degrees, arm joints) as an arm-only LeRobot goal."""
    goal = joint_map.to_lerobot(np.append(np.deg2rad(waypoint_sim_deg), 0.0))
    return {pos_key(j): goal[pos_key(j)] for j in joint_map.joints[:-1]}


class ApproachOffsets(Protocol):
    """Per-joint offset (LeRobot degrees) from a goal to its "from above" approach point."""

    def __call__(self, goal: Mapping[str, float]) -> dict[str, float]:
        """Offsets keyed like `goal`; the gripper's is always 0."""
        ...


class UniformOffsets:
    """Every arm joint offset by the same amount in the positive LeRobot direction."""

    def __init__(self, offset_deg: float, joints: tuple[str, ...]) -> None:
        self._offset = offset_deg
        self._gripper = pos_key(joints[-1])

    def __call__(self, goal: Mapping[str, float]) -> dict[str, float]:
        """`offset_deg` on every arm joint, 0 on the gripper."""
        return {k: 0.0 if k == self._gripper else self._offset for k in goal}


class GravityAwareOffsets:
    """ "Above" is the side gravity pulls away from, so the approach point lifts the links past each joint.

    A joint holding less than `min_torque_nm` (pan, roll, an arm balanced over
    its axis) uses the positive LeRobot side.
    """

    def __init__(
        self,
        joint_map: JointMap,
        gravity: GravityTorqueModel,
        offset_deg: float,
        min_torque_nm: float = SagCheckConfig().gravity_torque_floor_nm,
    ) -> None:
        self._map = joint_map
        self._gravity = gravity
        self._offset = offset_deg
        self._min_torque = min_torque_nm
        self._lerobot_sign = np.sign(joint_map.arm_delta_to_sim(np.ones(len(joint_map.joints) - 1)))

    def __call__(self, goal: Mapping[str, float]) -> dict[str, float]:
        """Offsets against the gravity torque at `goal`, in LeRobot degrees."""
        tau = self._gravity.torque(self._map.to_sim(goal))
        up_sim = np.where(np.abs(tau) > self._min_torque, np.sign(tau), self._lerobot_sign)
        arm = up_sim * self._lerobot_sign * self._offset
        keys = [pos_key(j) for j in self._map.joints]
        return {k: float(v) for k, v in zip(keys, [*arm, 0.0], strict=True)}


class RecordedOffsets:
    """Replays the "from above" offsets a recorded session used at each goal; older sessions fall back to uniform."""

    def __init__(self, session: SagSession, fallback: ApproachOffsets) -> None:
        self._by_goal = {
            s.goal_deg: dict(zip((pos_key(j) for j in session.joints), s.approach_offset_deg, strict=True))
            for s in session.samples
            if s.approach == "above" and s.approach_offset_deg is not None
        }
        self._joints = session.joints
        self._fallback = fallback

    def __call__(self, goal: Mapping[str, float]) -> dict[str, float]:
        """The recorded offsets for this goal, or the fallback's."""
        key = tuple(goal[pos_key(j)] for j in self._joints)
        return self._by_goal.get(key) or self._fallback(goal)


class SagCheckRunner:
    """Drives a `JointBus` through the sag check in a fixed pose order.

    Each pose is entered through `waypoint` (if given), then reached from its
    "above" approach point and its "below" one (`offsets`, mirrored). Every
    move longer than the final step is ramped at `ramp_speed_deg_s`. A joint
    that lags the ramp by more than LeRobot's relative-target cap gets its
    final goal clipped, so the final goal is re-sent until it goes through
    unclipped (up to `final_goal_timeout_s`); a sample is flagged `clamped`
    only if it never does. `plan` lists the same ramp targets for the
    pre-flight table check.
    """

    def __init__(
        self,
        bus: JointBus,
        clock: Clock,
        cfg: SagCheckConfig,
        joints: tuple[str, ...],
        offsets: ApproachOffsets | None = None,
        waypoint: Mapping[str, float] | None = None,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._cfg = cfg
        self._joints = joints
        self._offsets = offsets or UniformOffsets(cfg.approach_offset_deg, joints)
        self._waypoint = dict(waypoint) if waypoint is not None else None
        self._last: dict[str, float] | None = None
        self.last_sent: dict[str, float] | None = None
        self.clamp_count = 0

    def goal_dict(self, positions_deg: tuple[float, ...]) -> dict[str, float]:
        """`<joint>.pos` goal from values in session joint order."""
        return {pos_key(j): float(v) for j, v in zip(self._joints, positions_deg, strict=True)}

    def session(self, robot_id: str, poses: tuple[RecordedPose, ...]) -> SagSession:
        """Read back the servo settings, then run every pose."""
        settings = self._bus.read_settings()
        return SagSession(robot_id, self._joints, settings, self.run(poses))

    def plan(self, poses: tuple[RecordedPose, ...], rest: Mapping[str, float]) -> list[dict[str, float]]:
        """Every ramp target `run` then `finish(rest)` will move through, in order."""
        targets: list[dict[str, float]] = []
        for pose in poses:
            goal = self.goal_dict(pose.positions_deg)
            targets += self._entry(goal)
            targets += [self.approach_point(goal, "above"), goal, self.approach_point(goal, "below"), goal]
        return [*targets, *self._entry(dict(rest)), dict(rest)]

    def run(self, poses: tuple[RecordedPose, ...]) -> tuple[SagSample, ...]:
        """Enable torque in place, then sample each pose from above and from below."""
        self.hold_present()
        return tuple(sample for pose in poses for sample in self._visit(pose))

    def hold_present(self) -> dict[str, float]:
        """Make the present position the goal before enabling torque, so nothing jumps; return it."""
        self._last = self._bus.read_positions()
        self._bus.send_positions(self._last)
        self._bus.set_torque(True)
        return dict(self._last)

    def finish(self, rest_goal: dict[str, float]) -> None:
        """Ramp through the waypoint to the rest pose, then turn torque off."""
        for target in self._entry(rest_goal):
            self.ramp(target)
        self.ramp(rest_goal)
        self._bus.set_torque(False)

    def ramp(self, goal: dict[str, float]) -> None:
        """Interpolate from the last goal to `goal` at the configured joint speed."""
        start = self._last if self._last is not None else self._bus.read_positions()
        step_deg = self._cfg.ramp_speed_deg_s * self._cfg.ramp_period_s
        n = max(1, math.ceil(max(abs(goal[k] - start[k]) for k in goal) / step_deg))
        for k in range(1, n):
            self._send({key: start[key] + (goal[key] - start[key]) * k / n for key in goal})
            self._clock.sleep(self._cfg.ramp_period_s)
        self._send(dict(goal))
        self._clock.sleep(self._cfg.ramp_period_s)

    def approach_point(self, goal: dict[str, float], approach: Approach) -> dict[str, float]:
        """Goal shifted by the approach offsets (mirrored for "below")."""
        offsets = self._offsets(goal)
        return {k: v + APPROACH_SIGN[approach] * offsets[k] for k, v in goal.items()}

    def waypoint_for(self, goal: Mapping[str, float]) -> dict[str, float]:
        """The waypoint's arm joints with `goal`'s gripper (requires a waypoint)."""
        if self._waypoint is None:
            raise ValueError("runner has no waypoint")
        gripper = pos_key(self._joints[-1])
        return {**self._waypoint, gripper: goal[gripper]}

    def send_final(self, goal: dict[str, float]) -> tuple[dict[str, float], bool]:
        """Re-send `goal` until the bus accepts it unclipped; return the last send and whether it was still clipped."""
        attempts = max(1, math.ceil(self._cfg.final_goal_timeout_s / self._cfg.ramp_period_s))
        for _ in range(attempts):
            sent = self._send(goal)
            if not self._clipped(sent, goal):
                return sent, False
            self._clock.sleep(self._cfg.ramp_period_s)
        return sent, True

    def average_reads(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Mean and standard deviation of `n_reads` position readings."""
        reads = []
        for _ in range(self._cfg.n_reads):
            reading = self._bus.read_positions()
            reads.append([reading[pos_key(j)] for j in self._joints])
            self._clock.sleep(self._cfg.read_interval_s)
        arr = np.array(reads)
        return tuple(arr.mean(axis=0).tolist()), tuple(arr.std(axis=0).tolist())

    def _entry(self, goal: Mapping[str, float]) -> list[dict[str, float]]:
        """The waypoint before `goal`, or nothing without one."""
        return [] if self._waypoint is None else [self.waypoint_for(goal)]

    def _visit(self, pose: RecordedPose) -> tuple[SagSample, ...]:
        """Enter through the waypoint, then sample from above and from below."""
        for target in self._entry(self.goal_dict(pose.positions_deg)):
            self.ramp(target)
        return tuple(self._sample(pose, approach) for approach in APPROACHES)

    def _sample(self, pose: RecordedPose, approach: Approach) -> SagSample:
        """Ramp to the approach point, step onto the pose, settle, and average."""
        goal = self.goal_dict(pose.positions_deg)
        start = self.approach_point(goal, approach)
        self.ramp(start)
        sent, clamped = self.send_final(goal)
        self._clock.sleep(self._cfg.settle_s)
        mean, std = self.average_reads()
        offset = tuple(start[pos_key(j)] - goal[pos_key(j)] for j in self._joints)
        sent_deg = tuple(sent[pos_key(j)] for j in self._joints)
        status = self._bus.read_status()
        return SagSample(pose.pose_id, approach, pose.positions_deg, mean, std, status, sent_deg, clamped, offset)

    def _clipped(self, sent: dict[str, float], goal: dict[str, float]) -> bool:
        """True if any joint was sent somewhere other than its goal."""
        return any(abs(sent[k] - v) > self._cfg.clamp_tolerance_deg for k, v in goal.items())

    def _send(self, goal: dict[str, float]) -> dict[str, float]:
        """Send a goal, remember it as the ramp origin, and count it if the bus clipped it."""
        sent = self._bus.send_positions(goal)
        if self._clipped(sent, goal):
            self.clamp_count += 1
        self._last = dict(goal)
        self.last_sent = sent
        return sent

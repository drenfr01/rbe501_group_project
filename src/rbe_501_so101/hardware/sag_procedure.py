"""Phase 0 procedure: approach each pose from above and below, settle, and average readings."""

import math

import numpy as np

from rbe_501_so101.config import SagCheckConfig
from rbe_501_so101.hardware.bus import JointBus
from rbe_501_so101.hardware.clock import Clock
from rbe_501_so101.hardware.joint_mapping import pos_key
from rbe_501_so101.hardware.sag import APPROACHES, Approach, RecordedPose, SagSample, SagSession

APPROACH_SIGN: dict[Approach, float] = {"above": 1.0, "below": -1.0}


class SagCheckRunner:
    """Drives a `JointBus` through the sag check in a fixed pose order.

    "From above" starts every arm joint `approach_offset_deg` past the goal
    (in the positive LeRobot direction) and steps down to it; "from below"
    mirrors that. The gripper is never offset. Every move longer than the
    final step is ramped at `ramp_speed_deg_s`. Every send is compared with what
    the bus reports it actually sent, so LeRobot's relative-target clipping is
    counted and flagged on the affected samples.
    """

    def __init__(self, bus: JointBus, clock: Clock, cfg: SagCheckConfig, joints: tuple[str, ...]) -> None:
        self._bus = bus
        self._clock = clock
        self._cfg = cfg
        self._joints = joints
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

    def run(self, poses: tuple[RecordedPose, ...]) -> tuple[SagSample, ...]:
        """Enable torque in place, then sample each pose from above and from below."""
        self.hold_present()
        return tuple(self._sample(pose, approach) for pose in poses for approach in APPROACHES)

    def hold_present(self) -> dict[str, float]:
        """Make the present position the goal before enabling torque, so nothing jumps; return it."""
        self._last = self._bus.read_positions()
        self._bus.send_positions(self._last)
        self._bus.set_torque(True)
        return dict(self._last)

    def finish(self, rest_goal: dict[str, float]) -> None:
        """Ramp to the rest pose, then turn torque off."""
        self.ramp(rest_goal)
        self._bus.set_torque(False)

    def ramp(self, goal: dict[str, float]) -> None:
        """Interpolate from the last goal to `goal` at the configured joint speed."""
        start = self._last if self._last is not None else self._bus.read_positions()
        step_deg = self._cfg.ramp_speed_deg_s * self._cfg.ramp_period_s
        n = max(1, math.ceil(max(abs(goal[k] - start[k]) for k in goal) / step_deg))
        for k in range(1, n + 1):
            self._send({key: start[key] + (goal[key] - start[key]) * k / n for key in goal})
            self._clock.sleep(self._cfg.ramp_period_s)

    def _sample(self, pose: RecordedPose, approach: Approach) -> SagSample:
        """Ramp to the approach point, step onto the pose, settle, and average."""
        goal = self.goal_dict(pose.positions_deg)
        clamps_before = self.clamp_count
        self.ramp(self._approach_point(goal, approach))
        sent = self._send(goal)
        self._clock.sleep(self._cfg.settle_s)
        mean, std = self.average_reads()
        sent_deg = tuple(sent[pos_key(j)] for j in self._joints)
        status = self._bus.read_status()
        clamped = self.clamp_count > clamps_before
        return SagSample(pose.pose_id, approach, pose.positions_deg, mean, std, status, sent_deg, clamped)

    def _approach_point(self, goal: dict[str, float], approach: Approach) -> dict[str, float]:
        """Goal shifted by the approach offset on every arm joint."""
        shift = APPROACH_SIGN[approach] * self._cfg.approach_offset_deg
        gripper = pos_key(self._joints[-1])
        return {k: v if k == gripper else v + shift for k, v in goal.items()}

    def average_reads(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Mean and standard deviation of `n_reads` position readings."""
        reads = []
        for _ in range(self._cfg.n_reads):
            reading = self._bus.read_positions()
            reads.append([reading[pos_key(j)] for j in self._joints])
            self._clock.sleep(self._cfg.read_interval_s)
        arr = np.array(reads)
        return tuple(arr.mean(axis=0).tolist()), tuple(arr.std(axis=0).tolist())

    def _send(self, goal: dict[str, float]) -> dict[str, float]:
        """Send a goal, remember it as the ramp origin, and count it if the bus clipped it."""
        sent = self._bus.send_positions(goal)
        if any(abs(sent[k] - v) > self._cfg.clamp_tolerance_deg for k, v in goal.items()):
            self.clamp_count += 1
        self._last = dict(goal)
        self.last_sent = sent
        return sent

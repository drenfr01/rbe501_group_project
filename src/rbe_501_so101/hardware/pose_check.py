"""Pose check: command poses defined in sim degrees, then compare readings and level measurements.

Checks the joint map end to end: if `sim = lerobot + offset` is right, the
physical links sit at the angles each `CheckPose.check` describes.
"""

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import numpy as np

from rbe_501_so101.config import CheckPose, PoseCheckConfig, SagCheckConfig
from rbe_501_so101.hardware.bus import JointBus
from rbe_501_so101.hardware.clock import Clock
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.safety import SafetyAbort, SafetyMonitor
from rbe_501_so101.hardware.sag_procedure import SagCheckRunner

MM_PER_M = 1000.0


@dataclass(frozen=True)
class PoseCheckResult:
    """One visited pose: goal and reading in LeRobot and sim degrees, plus the operator's measurement."""

    name: str
    check: str
    sim_goal_deg: tuple[float, ...]
    lerobot_goal_deg: tuple[float, ...]
    sent_deg: tuple[float, ...]
    measured_deg: tuple[float, ...]
    measured_std_deg: tuple[float, ...]
    measured_sim_deg: tuple[float, ...]
    clamped: bool
    operator_note: str = ""
    recorded_at: str = ""

    def with_note(self, note: str) -> Self:
        """Copy with the operator's measurement attached."""
        return replace(self, operator_note=note)

    @classmethod
    def from_dict(cls, row: dict) -> Self:
        """Inverse of `dataclasses.asdict` (lists back to tuples)."""
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in row.items()})


class PoseCheckRepository:
    """JSON list of pose-check results, rewritten after every pose so an abort keeps the data."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, results: tuple[PoseCheckResult, ...]) -> Path:
        """Write all results."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(r) for r in results], indent=2))
        return self.path

    def load(self) -> tuple[PoseCheckResult, ...]:
        """All saved results."""
        return tuple(PoseCheckResult.from_dict(r) for r in json.loads(self.path.read_text()))


def segment_points(start: np.ndarray, end: np.ndarray, step_rad: float) -> np.ndarray:
    """Joint-space line from `start` to `end`, no joint moving more than `step_rad` per point."""
    n = max(1, math.ceil(float(np.abs(end - start).max()) / step_rad))
    return start + np.outer(np.linspace(0.0, 1.0, n + 1), end - start)


def plan_path(start: np.ndarray, poses: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    """Waypoints: the start, every pose in order, then the start again."""
    return [start, *poses, start]


def preflight(safety: SafetyMonitor, path: list[np.ndarray], cfg: PoseCheckConfig) -> float:
    """Smallest table margin (m) along the path, with and without sag; raise if it goes too low.

    A start already inside the clearance (an arm resting low) is allowed, but
    nothing on the path may go lower than the start does.
    """
    sag = np.deg2rad(np.append(cfg.sag_allowance_deg, 0.0))
    floor = min(0.0, safety.clearance_margin(path[0]), safety.clearance_margin(path[0] + sag))
    worst = math.inf
    for start, end in zip(path[:-1], path[1:], strict=True):
        for q in segment_points(start, end, math.radians(cfg.path_step_deg)):
            worst = min(worst, safety.clearance_margin(q), safety.clearance_margin(q + sag))
    if worst < floor:
        raise SafetyAbort(f"path comes {MM_PER_M * -worst:.0f} mm inside the table clearance")
    return worst


def _driver_config(cfg: PoseCheckConfig) -> SagCheckConfig:
    """Ramp, settle, and averaging settings for the shared driver."""
    return SagCheckConfig(
        settle_s=cfg.settle_s,
        n_reads=cfg.n_reads,
        read_interval_s=cfg.read_interval_s,
        ramp_speed_deg_s=cfg.ramp_speed_deg_s,
        ramp_period_s=cfg.ramp_period_s,
        clamp_tolerance_deg=cfg.clamp_tolerance_deg,
    )


class PoseCheckRunner:
    """Ramps a `JointBus` to each `CheckPose`, settles, and records the averaged reading.

    The gripper keeps its starting reading throughout; only the arm is commanded.
    """

    def __init__(self, bus: JointBus, clock: Clock, joint_map: JointMap, cfg: PoseCheckConfig, joints: tuple[str, ...]) -> None:
        self._driver = SagCheckRunner(bus, clock, _driver_config(cfg), joints)
        self._clock = clock
        self._map = joint_map
        self._cfg = cfg
        self._joints = joints
        self._start: dict[str, float] = {}

    def start(self) -> dict[str, float]:
        """Hold the present pose with torque on; it is also where `finish` returns."""
        self._start = self._driver.hold_present()
        return self._start

    def lerobot_goal(self, pose: CheckPose) -> dict[str, float]:
        """LeRobot goal for a pose, with the gripper held at its starting reading."""
        goal = self._map.to_lerobot(np.append(np.deg2rad(pose.sim_deg), 0.0))
        gripper = pos_key(self._joints[-1])
        goal[gripper] = self._start.get(gripper, goal[gripper])
        return goal

    def visit(self, pose: CheckPose) -> PoseCheckResult:
        """Ramp to the pose, settle, and average the readings."""
        goal = self.lerobot_goal(pose)
        clamps_before = self._driver.clamp_count
        self._driver.ramp(goal)
        self._clock.sleep(self._cfg.settle_s)
        mean, std = self._driver.average_reads()
        return self._result(pose, goal, mean, std, self._driver.clamp_count > clamps_before)

    def finish(self) -> None:
        """Ramp back to the starting pose, then turn torque off."""
        self._driver.finish(self._start)

    def _result(self, pose: CheckPose, goal: dict[str, float], mean: tuple[float, ...], std: tuple[float, ...], clamped: bool) -> PoseCheckResult:
        """Bundle goal, sent command, and reading in both unit systems."""
        sent = self._driver.last_sent or goal
        measured_sim = np.rad2deg(self._map.to_sim({pos_key(j): v for j, v in zip(self._joints, mean, strict=True)}))
        return PoseCheckResult(
            name=pose.name,
            check=pose.check,
            sim_goal_deg=tuple(pose.sim_deg),
            lerobot_goal_deg=tuple(goal[pos_key(j)] for j in self._joints),
            sent_deg=tuple(sent[pos_key(j)] for j in self._joints),
            measured_deg=mean,
            measured_std_deg=std,
            measured_sim_deg=tuple(float(v) for v in measured_sim[: len(pose.sim_deg)]),
            clamped=clamped,
            recorded_at=datetime.now(UTC).isoformat(),
        )

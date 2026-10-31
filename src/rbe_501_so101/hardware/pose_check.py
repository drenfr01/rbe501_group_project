"""Pose check: command poses defined in sim degrees, then compare readings and level measurements.

Checks the joint map end to end: if `sim = lerobot + offset` is right, the
physical links sit at the angles each `CheckPose.check` describes.
"""

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import numpy as np

from rbe_501_so101.config import CheckPose, PoseCheckConfig, SagCheckConfig
from rbe_501_so101.hardware.bus import JointBus
from rbe_501_so101.hardware.clock import Clock
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.sag_procedure import SagCheckRunner
from rbe_501_so101.hardware.types import MotorSettings

SETTINGS_FILENAME = "servo_settings.json"


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

    @property
    def settings_path(self) -> Path:
        """Servo registers read back at the start of the run, next to the results."""
        return self.path.with_name(SETTINGS_FILENAME)

    def save_settings(self, settings: tuple[MotorSettings, ...]) -> Path:
        """Write the read-back servo settings (gain, torque limits) for this run."""
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps([asdict(s) for s in settings], indent=2))
        return self.settings_path

    def load_settings(self) -> tuple[MotorSettings, ...]:
        """Servo settings saved with this run."""
        return tuple(MotorSettings(**s) for s in json.loads(self.settings_path.read_text()))


def _driver_config(cfg: PoseCheckConfig) -> SagCheckConfig:
    """Ramp, settle, and averaging settings for the shared driver."""
    return SagCheckConfig(
        settle_s=cfg.settle_s,
        n_reads=cfg.n_reads,
        read_interval_s=cfg.read_interval_s,
        ramp_speed_deg_s=cfg.ramp_speed_deg_s,
        ramp_period_s=cfg.ramp_period_s,
        clamp_tolerance_deg=cfg.clamp_tolerance_deg,
        final_goal_timeout_s=cfg.final_goal_timeout_s,
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
        self._driver.ramp(goal)
        _, clamped = self._driver.send_final(goal)
        self._clock.sleep(self._cfg.settle_s)
        mean, std = self._driver.average_reads()
        return self._result(pose, goal, mean, std, clamped)

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

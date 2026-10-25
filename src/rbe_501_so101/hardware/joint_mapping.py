"""Convert LeRobot joint readings (degrees, gripper 0-100) to sim radians and back."""

from collections.abc import Mapping
from typing import Self

import mujoco
import numpy as np

from rbe_501_so101.config import HardwareConfig, JointMappingEntry, SimConfig
from rbe_501_so101.envs.types import JointLimits

POS_SUFFIX = ".pos"


def pos_key(joint: str) -> str:
    """LeRobot observation/action key for one joint."""
    return f"{joint}{POS_SUFFIX}"


class JointMap:
    """Per-joint `sim = sign * deg2rad(lerobot) + offset`, plus a linear gripper map.

    Joint vectors follow `SimConfig.joints` order: arm joints, then the gripper.
    """

    def __init__(
        self,
        arm_joints: tuple[str, ...],
        gripper_joint: str,
        entries: tuple[JointMappingEntry, ...],
        gripper_percent_range: tuple[float, float],
        gripper_range_rad: tuple[float, float],
    ) -> None:
        if len(entries) != len(arm_joints):
            raise ValueError(f"need {len(arm_joints)} joint map entries, got {len(entries)}")
        self._arm = arm_joints
        self._gripper = gripper_joint
        self._sign = np.array([e.sign for e in entries], dtype=float)
        self._offset = np.array([e.offset_rad for e in entries], dtype=float)
        self._pct = gripper_percent_range
        self._grip_rad = gripper_range_rad

    @classmethod
    def from_model(cls, model: mujoco.MjModel, sim_cfg: SimConfig, hw_cfg: HardwareConfig) -> Self:
        """Build a map whose gripper range is read from the MuJoCo model."""
        low, high = model.joint(sim_cfg.gripper_joint).range
        return cls(sim_cfg.arm_joints, sim_cfg.gripper_joint, hw_cfg.joint_map, hw_cfg.gripper_percent_range, (low, high))

    @property
    def joints(self) -> tuple[str, ...]:
        """Joint names in vector order."""
        return (*self._arm, self._gripper)

    def to_sim(self, reading: Mapping[str, float]) -> np.ndarray:
        """LeRobot `<joint>.pos` readings to sim radians (arm joints, then gripper)."""
        values = self._ordered(reading)
        arm = self._sign * np.deg2rad(values[:-1]) + self._offset
        return np.append(arm, self._percent_to_rad(values[-1]))

    def to_lerobot(self, q_rad: np.ndarray) -> dict[str, float]:
        """Exact inverse of `to_sim`, keyed the way `send_action` expects."""
        q = np.asarray(q_rad, dtype=float)
        arm = np.rad2deg((q[: len(self._arm)] - self._offset) / self._sign)
        values = [*arm, self._rad_to_percent(float(q[len(self._arm)]))]
        return {pos_key(name): float(v) for name, v in zip(self.joints, values, strict=True)}

    def arm_delta_to_sim(self, delta_deg: np.ndarray) -> np.ndarray:
        """Convert a LeRobot-degree difference (no offset) on the arm joints to sim radians."""
        return self._sign * np.deg2rad(np.asarray(delta_deg, dtype=float))

    def arm_limits_to_sim(self, limits_deg: Mapping[str, tuple[float, float]]) -> JointLimits:
        """LeRobot arm limits (degrees) to sim limits, re-ordered after any sign flip."""
        low = self._sign * np.deg2rad([limits_deg[n][0] for n in self._arm]) + self._offset
        high = self._sign * np.deg2rad([limits_deg[n][1] for n in self._arm]) + self._offset
        return JointLimits(lower=np.minimum(low, high), upper=np.maximum(low, high))

    def _ordered(self, reading: Mapping[str, float]) -> np.ndarray:
        """Readings in joint order; raise naming every missing key."""
        missing = [pos_key(n) for n in self.joints if pos_key(n) not in reading]
        if missing:
            raise KeyError(f"reading is missing joints: {', '.join(missing)}")
        return np.array([reading[pos_key(n)] for n in self.joints], dtype=float)

    def _percent_to_rad(self, percent: float) -> float:
        """Linear map from the gripper percent range onto the sim gripper range."""
        return float(np.interp(percent, self._pct, self._grip_rad))

    def _rad_to_percent(self, angle: float) -> float:
        """Inverse of `_percent_to_rad`."""
        return float(np.interp(angle, self._grip_rad, self._pct))

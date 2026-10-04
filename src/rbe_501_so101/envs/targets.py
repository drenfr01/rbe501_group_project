"""Reach target samplers that only produce kinematically reachable positions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import mujoco
import numpy as np

from rbe_501_so101.config import SimConfig, TargetSamplerConfig


@dataclass(frozen=True)
class ReachabilityMap:
    """Forward-kinematics samples of the gripper tip and whether each passes the filters."""

    points: np.ndarray
    radial_m: np.ndarray
    accepted: np.ndarray

    @property
    def acceptance_rate(self) -> float:
        """Fraction of sampled configurations that pass the clearance and radial filters."""
        return float(np.mean(self.accepted))

    def save(self, path: Path) -> None:
        """Write the map to a compressed .npz file."""
        np.savez_compressed(path, points=self.points, radial_m=self.radial_m, accepted=self.accepted)


class ReachableTargetSampler:
    """Samples targets from random arm configurations via forward kinematics.

    Joint angles are drawn uniformly within the limits shrunk by a margin; the
    resulting gripper-site position is accepted only if it clears the table
    (z >= z_min) and lies within the radial band around the shoulder-pan axis.
    Every target is therefore reachable by construction.
    """

    def __init__(self, model: mujoco.MjModel, sim_cfg: SimConfig, cfg: TargetSamplerConfig) -> None:
        self._model = model
        self._cfg = cfg
        self._data = mujoco.MjData(model)
        joint_ids = [model.joint(n).id for n in sim_cfg.joints]
        n_arm = len(sim_cfg.arm_joints)
        self._arm_qpos_adr = model.jnt_qposadr[joint_ids[:n_arm]]
        self._gripper_qpos_adr = model.jnt_qposadr[joint_ids[n_arm]]
        self._gripper_angle = float(np.deg2rad(sim_cfg.initial_deg[n_arm]))
        arm_range = model.jnt_range[joint_ids[:n_arm]]
        self._low = arm_range[:, 0] + cfg.joint_limit_margin_rad
        self._high = arm_range[:, 1] - cfg.joint_limit_margin_rad
        self._site_id = model.site(sim_cfg.tcp_site).id
        self._axis_xy = self._radial_axis_xy(model, sim_cfg.radial_axis_joint)

    @classmethod
    def from_config(cls, sim_cfg: SimConfig, cfg: TargetSamplerConfig) -> Self:
        """Build a sampler with its own copy of the configured model."""
        return cls(mujoco.MjModel.from_xml_path(str(sim_cfg.model_path)), sim_cfg, cfg)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Return a reachable world-frame XYZ target in meters."""
        for _ in range(self._cfg.max_attempts):
            point = self._random_tip(rng)
            if self._accepts(point):
                return point
        raise RuntimeError(f"No target accepted after {self._cfg.max_attempts} attempts; check the radial band")

    def radial_distance(self, xyz: np.ndarray) -> float:
        """Horizontal distance (m) from the shoulder-pan axis to a world point."""
        return float(np.linalg.norm(np.asarray(xyz)[:2] - self._axis_xy))

    def reachability_map(self, rng: np.random.Generator, n_samples: int) -> ReachabilityMap:
        """Sample `n_samples` configurations and record which tips pass the filters."""
        points = np.array([self._random_tip(rng) for _ in range(n_samples)])
        radial = np.linalg.norm(points[:, :2] - self._axis_xy, axis=1)
        return ReachabilityMap(points=points, radial_m=radial, accepted=self._accepts_many(points, radial))

    def _random_tip(self, rng: np.random.Generator) -> np.ndarray:
        """Gripper-site position for a uniformly random arm configuration."""
        self._data.qpos[self._arm_qpos_adr] = rng.uniform(self._low, self._high)
        self._data.qpos[self._gripper_qpos_adr] = self._gripper_angle
        mujoco.mj_kinematics(self._model, self._data)
        return self._data.site_xpos[self._site_id].copy()

    def _accepts(self, point: np.ndarray) -> bool:
        """Clearance and radial-band filter for a single point."""
        return bool(self._accepts_many(point[None, :], np.array([self.radial_distance(point)]))[0])

    def _accepts_many(self, points: np.ndarray, radial: np.ndarray) -> np.ndarray:
        """Vectorized clearance and radial-band filter."""
        in_band = (radial >= self._cfg.radial_min_m) & (radial <= self._cfg.radial_max_m)
        return (points[:, 2] >= self._cfg.z_min_m) & in_band

    @staticmethod
    def _radial_axis_xy(model: mujoco.MjModel, joint_name: str) -> np.ndarray:
        """World XY of a vertical joint axis at the model's reference pose."""
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        return data.xanchor[model.joint(joint_name).id][:2].copy()

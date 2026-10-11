"""Reach target samplers that only produce kinematically reachable positions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import mujoco
import numpy as np

from rbe_501_so101.config import ReachTaskConfig, SimConfig, TargetSamplerConfig
from rbe_501_so101.envs.model_indices import (
    ModelIndices,
    descendant_body_ids,
    disable_body_collisions,
    has_robot_world_contact,
    park_body,
    table_clearance_z,
)


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
    resulting gripper-site position is accepted only if it clears the table,
    lies within the radial band around the shoulder-pan axis, and the pose
    does not put the arm in contact with the world. Every accepted target is
    therefore reachable in free space by construction.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ids: ModelIndices,
        sim_cfg: SimConfig,
        cfg: TargetSamplerConfig,
        gripper_hold_deg: float | None = None,
    ) -> None:
        self._model = model
        self._ids = ids
        self._cfg = cfg
        self._data = mujoco.MjData(model)
        hold_deg = ReachTaskConfig().gripper_hold_deg if gripper_hold_deg is None else gripper_hold_deg
        self._gripper_angle = float(np.deg2rad(hold_deg))
        arm_range = model.jnt_range[ids.arm_joint_ids]
        self._low = arm_range[:, 0] + cfg.joint_limit_margin_rad
        self._high = arm_range[:, 1] - cfg.joint_limit_margin_rad
        self._axis_xy = self._radial_axis_xy(model, sim_cfg.radial_axis_joint)
        self._z_min_m = table_clearance_z(model, sim_cfg.table_top_geom, cfg.clearance_m)
        self._robot_bodies = descendant_body_ids(model, sim_cfg.robot_root_body)
        self._prepare_scene(sim_cfg)

    @classmethod
    def from_config(
        cls,
        sim_cfg: SimConfig,
        cfg: TargetSamplerConfig,
        gripper_hold_deg: float | None = None,
    ) -> Self:
        """Build a sampler with its own copy of the configured model."""
        model = mujoco.MjModel.from_xml_path(str(sim_cfg.model_path))
        return cls(model, ModelIndices.from_model(model, sim_cfg), sim_cfg, cfg, gripper_hold_deg)

    @property
    def z_min_m(self) -> float:
        """World-Z threshold: table-top surface plus configured clearance."""
        return self._z_min_m

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

    def has_world_contact(self) -> bool:
        """True if the scratch configuration contacts a non-robot body."""
        return has_robot_world_contact(self._model, self._data, self._robot_bodies)

    def reachability_map(self, rng: np.random.Generator, n_samples: int) -> ReachabilityMap:
        """Sample `n_samples` configurations and record which tips pass the filters."""
        points = np.empty((n_samples, 3))
        radial = np.empty(n_samples)
        accepted = np.empty(n_samples, dtype=bool)
        for i in range(n_samples):
            points[i] = self._random_tip(rng)
            radial[i] = self.radial_distance(points[i])
            accepted[i] = self._accepts(points[i])
        return ReachabilityMap(points=points, radial_m=radial, accepted=accepted)

    def _random_tip(self, rng: np.random.Generator) -> np.ndarray:
        """Gripper-site position for a uniformly random arm configuration."""
        self._data.qpos[self._ids.arm_qpos_adr] = rng.uniform(self._low, self._high)
        self._data.qpos[self._ids.gripper_qpos_adr] = self._gripper_angle
        mujoco.mj_kinematics(self._model, self._data)
        return self._data.site_xpos[self._ids.tcp_site].copy()

    def _accepts(self, point: np.ndarray) -> bool:
        """Clearance, radial-band, and collision-free filter for the current pose."""
        return self._in_workspace(point) and not self.has_world_contact()

    def _in_workspace(self, point: np.ndarray) -> bool:
        """Clearance and radial-band filter for a single point."""
        radial = self.radial_distance(point)
        in_band = self._cfg.radial_min_m <= radial <= self._cfg.radial_max_m
        return bool(point[2] >= self._z_min_m and in_band)

    def _prepare_scene(self, sim_cfg: SimConfig) -> None:
        """Disable leftover grasp objects so they cannot block reach samples."""
        for name in sim_cfg.parked_bodies:
            disable_body_collisions(self._model, name)
            park_body(self._model, self._data, name)

    @staticmethod
    def _radial_axis_xy(model: mujoco.MjModel, joint_name: str) -> np.ndarray:
        """World XY of a vertical joint axis at the model's reference pose."""
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        return data.xanchor[model.joint(joint_name).id][:2].copy()

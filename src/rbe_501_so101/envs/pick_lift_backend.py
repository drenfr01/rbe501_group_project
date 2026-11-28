"""MuJoCo backend for pick-and-lift: a free cube, a gripper command, and grasp sensing."""

import mujoco
import numpy as np

from rbe_501_so101.config import (
    ContactSolverConfig,
    DynamicsRandomizationConfig,
    GripperConfig,
    ObjectConfig,
    RenderConfig,
    SimConfig,
)
from rbe_501_so101.envs.grasp_model import FIXED_PAD_GEOM, MOVING_PAD_GEOM, build_grasp_model
from rbe_501_so101.envs.model_indices import table_clearance_z
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend, RenderMode
from rbe_501_so101.envs.types import GraspState, ObjectState

FREE_QPOS_DIM = 7
FREE_DOF_DIM = 6
POSITION = slice(0, 3)
QUATERNION = slice(3, 7)
SOLID_CUBE_INERTIA_DIVISOR = 6.0


def cube_inertia(mass_kg: float, side_m: float) -> np.ndarray:
    """Principal inertia of a solid cube: I = m s² / 6 about each axis."""
    return np.full(3, mass_kg * side_m**2 / SOLID_CUBE_INERTIA_DIVISOR)


def joint_axis_xy(model: mujoco.MjModel, joint_name: str) -> np.ndarray:
    """World XY of a joint axis anchor at the model's reference pose."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return data.xanchor[model.joint(joint_name).id][:2].copy()


def yaw_quaternion(yaw_rad: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation by `yaw_rad` about z_s."""
    half = yaw_rad / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


class MujocoPickLiftBackend(MujocoSo101Backend):
    """MuJoCo backend with cube placement, gripper control, and grasp sensing."""

    def __init__(
        self,
        sim_cfg: SimConfig = SimConfig(parked_bodies=()),
        gripper_cfg: GripperConfig = GripperConfig(),
        object_cfg: ObjectConfig = ObjectConfig(),
        contact_cfg: ContactSolverConfig = ContactSolverConfig(),
        render_mode: RenderMode | None = None,
        render_cfg: RenderConfig = RenderConfig(),
    ) -> None:
        model = build_grasp_model(sim_cfg, gripper_cfg, object_cfg, contact_cfg)
        super().__init__(sim_cfg, DynamicsRandomizationConfig(), render_mode, render_cfg, model=model)
        self._object_cfg = object_cfg
        self._p_b = np.asarray(gripper_cfg.pinch_point_b(object_cfg.side_m))
        self._scratch = mujoco.MjData(self.model)
        self._cache_object_addresses(object_cfg)
        self._pads = frozenset(self.model.geom(name).id for name in (FIXED_PAD_GEOM, MOVING_PAD_GEOM))
        self._gripper_range = self.model.jnt_range[self.ids.joint_ids[self.ids.n_arm]].copy()
        self._rest_z = table_clearance_z(self.model, sim_cfg.table_top_geom, object_cfg.side_m / 2)
        self._pan_axis_xy = joint_axis_xy(self.model, sim_cfg.radial_axis_joint)

    @property
    def object_mass_kg(self) -> float:
        """Current mass of the object body."""
        return float(self.model.body_mass[self._obj_body])

    @property
    def rest_z(self) -> float:
        """World Z of the object center when it rests on the table."""
        return self._rest_z

    @property
    def pan_axis_xy(self) -> np.ndarray:
        """World XY of the shoulder-pan axis."""
        return self._pan_axis_xy

    def place_object(self, p_s: np.ndarray, yaw_rad: float, mass_kg: float) -> None:
        """Set the object's mass, then put it at rest at p_s with the given yaw."""
        self._set_mass(mass_kg)
        qpos = self.data.qpos[self._obj_qpos : self._obj_qpos + FREE_QPOS_DIM]
        qpos[POSITION] = p_s
        qpos[QUATERNION] = yaw_quaternion(yaw_rad)
        self.data.qvel[self._obj_dof : self._obj_dof + FREE_DOF_DIM] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def apply_gripper_delta(self, delta_rad: float) -> None:
        """Integrate a relative gripper command and clip it to the jaw range."""
        n = self.ids.n_arm
        self._q_cmd[n] = np.clip(self._q_cmd[n] + delta_rad, *self._gripper_range)

    def grasp_state(self) -> GraspState:
        """Robot state, pinch point in {s}, object state, and two-pad contact."""
        robot = self._state()
        R_sb = self.data.site_xmat[self.ids.tcp_site].reshape(3, 3)
        p_pinch = robot.tcp_pos + R_sb @ self._p_b
        p_obj = self.data.xpos[self._obj_body].copy()
        obj = ObjectState(p_s=p_obj, rest_z=self._rest_z, mass_kg=self.object_mass_kg)
        return GraspState(robot=robot, p_pinch=p_pinch, obj=obj, grasped=self._pads <= self._geoms_touching_object())

    def _cache_object_addresses(self, object_cfg: ObjectConfig) -> None:
        """Body, geom, and free-joint addresses of the object."""
        self._obj_body = self.model.body(object_cfg.body).id
        self._obj_geom = self.model.geom(object_cfg.geom).id
        joint = int(self.model.body_jntadr[self._obj_body])
        self._obj_qpos = int(self.model.jnt_qposadr[joint])
        self._obj_dof = int(self.model.jnt_dofadr[joint])

    def _set_mass(self, mass_kg: float) -> None:
        """Update cube mass, inertia, and MuJoCo's derived constants."""
        if mass_kg == self.object_mass_kg:
            return
        self.model.body_mass[self._obj_body] = mass_kg
        self.model.body_inertia[self._obj_body] = cube_inertia(mass_kg, self._object_cfg.side_m)
        mujoco.mj_setConst(self.model, self._scratch)

    def _geoms_touching_object(self) -> set[int]:
        """IDs of every geom currently in contact with the object."""
        touching: set[int] = set()
        for contact in self.data.contact[: self.data.ncon]:
            if contact.geom1 == self._obj_geom:
                touching.add(int(contact.geom2))
            elif contact.geom2 == self._obj_geom:
                touching.add(int(contact.geom1))
        return touching

"""Shared MuJoCo name-to-index lookups and contact helpers."""

from dataclasses import dataclass
from typing import Self

import mujoco
import numpy as np

from rbe_501_so101.config import SimConfig

PARKED_POSITION = np.array([100.0, 100.0, -1.0, 1.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class ModelIndices:
    """Pre-cached MuJoCo addresses so hot paths never do string lookups."""

    n_arm: int
    joint_ids: np.ndarray
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    actuator_ids: np.ndarray
    tcp_site: int
    target_mocap: int
    payload_body: int

    @classmethod
    def from_model(cls, model: mujoco.MjModel, cfg: SimConfig) -> Self:
        """Resolve all configured names to integer IDs and addresses."""
        joint_ids = np.array([model.joint(n).id for n in cfg.joints])
        return cls(
            n_arm=len(cfg.arm_joints),
            joint_ids=joint_ids,
            qpos_adr=model.jnt_qposadr[joint_ids].copy(),
            dof_adr=model.jnt_dofadr[joint_ids].copy(),
            actuator_ids=np.array([model.actuator(n + cfg.actuator_suffix).id for n in cfg.joints]),
            tcp_site=model.site(cfg.tcp_site).id,
            target_mocap=int(model.body(cfg.target_mocap_body).mocapid[0]),
            payload_body=model.body(cfg.payload_body).id,
        )

    @property
    def arm_qpos_adr(self) -> np.ndarray:
        """qpos addresses of the arm joints."""
        return self.qpos_adr[: self.n_arm]

    @property
    def gripper_qpos_adr(self) -> int:
        """qpos address of the gripper joint."""
        return int(self.qpos_adr[self.n_arm])

    @property
    def arm_dof_adr(self) -> np.ndarray:
        """dof addresses of the arm joints."""
        return self.dof_adr[: self.n_arm]

    @property
    def arm_joint_ids(self) -> np.ndarray:
        """MuJoCo joint IDs of the arm joints."""
        return self.joint_ids[: self.n_arm]


def descendant_body_ids(model: mujoco.MjModel, root_name: str) -> frozenset[int]:
    """Body IDs of `root_name` and every body under it."""
    root = model.body(root_name).id
    return frozenset(b for b in range(model.nbody) if _is_descendant(model, b, root))


def has_robot_world_contact(
    model: mujoco.MjModel, data: mujoco.MjData, robot_bodies: frozenset[int]
) -> bool:
    """True if any contact pairs a robot body with a non-robot body."""
    mujoco.mj_collision(model, data)
    for contact in data.contact[: data.ncon]:
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if bodies & robot_bodies and bodies - robot_bodies:
            return True
    return False


def disable_body_collisions(model: mujoco.MjModel, body_name: str) -> None:
    """Turn off contacts and hide every geom attached to `body_name`."""
    body_id = model.body(body_name).id
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) != body_id:
            continue
        model.geom_contype[geom] = 0
        model.geom_conaffinity[geom] = 0
        model.geom_rgba[geom, 3] = 0.0


def park_body(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> None:
    """Move a freejoint body far from the workspace so it cannot interact."""
    body_id = model.body(body_name).id
    joint = int(model.body_jntadr[body_id])
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        return
    adr = int(model.jnt_qposadr[joint])
    data.qpos[adr : adr + 7] = PARKED_POSITION


def table_clearance_z(model: mujoco.MjModel, geom_name: str, clearance_m: float) -> float:
    """World-Z of the table-top surface plus a vertical clearance."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    geom_id = model.geom(geom_name).id
    top = float(data.geom_xpos[geom_id][2] + model.geom_size[geom_id][2])
    return top + clearance_m


def _is_descendant(model: mujoco.MjModel, body: int, root: int) -> bool:
    """True if `body` is `root` or nested under it."""
    while body not in (0, root):
        body = int(model.body_parentid[body])
    return body == root

"""Build the MuJoCo grasp model from the SO-101 scene."""

from dataclasses import dataclass

import mujoco
import numpy as np

from rbe_501_so101.config import ContactSolverConfig, GripperConfig, ObjectConfig, SimConfig

FIXED_PAD_GEOM = "fixed_finger_pad"
MOVING_PAD_GEOM = "moving_finger_pad"
COLLISION_ONLY_GROUP = 3
INVISIBLE_RGBA = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class FingerHull:
    """Distal finger vertices (in the jaw geom's frame) and where that geom sits on its body."""

    geom_name: str
    vertices: np.ndarray
    body: str
    pos: np.ndarray
    quat: np.ndarray


def build_grasp_model(
    sim_cfg: SimConfig,
    gripper_cfg: GripperConfig,
    object_cfg: ObjectConfig,
    contact_cfg: ContactSolverConfig = ContactSolverConfig(),
) -> mujoco.MjModel:
    """Compile finger collision hulls and a cube of the configured size."""
    hulls = finger_hulls(sim_cfg, gripper_cfg)
    spec = mujoco.MjSpec.from_file(str(sim_cfg.model_path))
    _make_visual_only(spec, {gripper_cfg.fixed_jaw_mesh, gripper_cfg.moving_jaw_mesh})
    for hull in hulls:
        _add_hull_geom(spec, hull, gripper_cfg)
    spec.geom(object_cfg.geom).size = [object_cfg.side_m / 2] * 3
    spec.site(sim_cfg.target_site).rgba = list(INVISIBLE_RGBA)
    _set_contact_solver(spec, contact_cfg)
    return spec.compile()


def finger_hulls(sim_cfg: SimConfig, cfg: GripperConfig) -> tuple[FingerHull, FingerHull]:
    """Fixed- and moving-jaw vertices whose approach coordinate z_b exceeds the cutoffs."""
    model = mujoco.MjModel.from_xml_path(str(sim_cfg.model_path))
    data = mujoco.MjData(model)
    data.qpos[model.joint(sim_cfg.gripper_joint).qposadr[0]] = 0.0
    mujoco.mj_kinematics(model, data)
    site = model.site(sim_cfg.tcp_site).id
    return (
        _distal_hull(model, data, site, cfg.fixed_jaw_mesh, cfg.fixed_pad_cutoff_m, FIXED_PAD_GEOM),
        _distal_hull(model, data, site, cfg.moving_jaw_mesh, cfg.moving_pad_cutoff_m, MOVING_PAD_GEOM),
    )


def _distal_hull(
    model: mujoco.MjModel, data: mujoco.MjData, site: int, mesh_name: str, cutoff_m: float, geom_name: str
) -> FingerHull:
    """Keep the mesh vertices that lie past `cutoff_m` along the approach axis z_b."""
    geom = _geom_using_mesh(model, mesh_name)
    mesh = model.mesh(mesh_name).id
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start : start + count]
    z_b = _geom_points_in_site_frame(data, geom, site, vertices)[:, 2]
    return FingerHull(
        geom_name=geom_name,
        vertices=vertices[z_b > cutoff_m].copy(),
        body=model.body(int(model.geom_bodyid[geom])).name,
        pos=model.geom_pos[geom].copy(),
        quat=model.geom_quat[geom].copy(),
    )


def _geom_points_in_site_frame(data: mujoco.MjData, geom: int, site: int, points: np.ndarray) -> np.ndarray:
    """Map geom-frame points to {b}: p_b = R_sb^T (R_sg p_g + p_sg - p_sb)."""
    R_sg = data.geom_xmat[geom].reshape(3, 3)
    p_s = data.geom_xpos[geom] + points @ R_sg.T
    R_sb = data.site_xmat[site].reshape(3, 3)
    return (p_s - data.site_xpos[site]) @ R_sb


def _set_contact_solver(spec: mujoco.MjSpec, cfg: ContactSolverConfig) -> None:
    """Friction cone and impratio for every contact in the scene."""
    if cfg.elliptic_cone:
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = cfg.impratio


def _geom_using_mesh(model: mujoco.MjModel, mesh_name: str) -> int:
    """ID of the geom that renders `mesh_name`."""
    mesh = model.mesh(mesh_name).id
    return next(g for g in range(model.ngeom) if model.geom_dataid[g] == mesh)


def _make_visual_only(spec: mujoco.MjSpec, mesh_names: set[str]) -> None:
    """Stop the whole-jaw meshes from colliding; they still render."""
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname in mesh_names:
            geom.contype = 0
            geom.conaffinity = 0


def _add_hull_geom(spec: mujoco.MjSpec, hull: FingerHull, cfg: GripperConfig) -> None:
    """Attach a massless, invisible collision geom whose convex hull is the distal finger."""
    mesh_name = f"{hull.geom_name}_hull"
    spec.add_mesh(name=mesh_name, uservert=hull.vertices.ravel().tolist())
    spec.body(hull.body).add_geom(
        name=hull.geom_name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        pos=hull.pos,
        quat=hull.quat,
        friction=list(cfg.pad_friction),
        solref=list(cfg.pad_solref),
        solimp=list(cfg.pad_solimp),
        density=0.0,
        group=COLLISION_ONLY_GROUP,
        rgba=list(INVISIBLE_RGBA),
    )

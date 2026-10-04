"""Physics tests: no gravity compensation, torque caps, sag direction/bounds, mocap target."""

import mujoco
import numpy as np

from rbe_501_so101.config import ReachTaskConfig, SimConfig

ROBOT_BODIES = (
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "moving_jaw_so101_v1_link",
)
JAW_GEOMS = ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1")
JAW_FRICTION = (1.5, 0.01, 0.001)
JAW_SOLREF = (0.005, 1.0)
JAW_SOLIMP = (0.9, 0.95, 0.001)
SETTLED_JOINT_SPEED_RAD_S = 0.01
MAX_SAG_M = 0.030
N_ARM = len(SimConfig().arm_joints)


def _geom_by_mesh(model: mujoco.MjModel, mesh_name: str) -> int:
    mesh_id = model.mesh(mesh_name).id
    return int(np.flatnonzero(model.geom_dataid == mesh_id)[0])


def _robot_bodies(model: mujoco.MjModel, root_name: str) -> set[int]:
    root = model.body(root_name).id
    return {b for b in range(model.nbody) if _is_descendant(model, b, root)}


def _is_descendant(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body not in (0, root):
        body = int(model.body_parentid[body])
    return body == root


def test_gravity_compensation_is_disabled(raw_model):
    for name in ROBOT_BODIES:
        assert raw_model.body_gravcomp[raw_model.body(name).id] == 0.0, name


def test_xml_torque_caps_equal_stall_torque(raw_model, sim_cfg):
    stall = sim_cfg.stall_torque_nm
    for name in sim_cfg.joints:
        np.testing.assert_allclose(raw_model.jnt_actfrcrange[raw_model.joint(name).id], [-stall, stall])


def test_backend_applies_configured_torque_caps():
    from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend

    cfg = SimConfig(stall_torque_nm=1.5)
    backend = MujocoSo101Backend(cfg)
    for name in cfg.arm_joints:
        np.testing.assert_allclose(backend.model.jnt_actfrcrange[backend.model.joint(name).id], [-1.5, 1.5])
    backend.close()


def test_jaw_geoms_have_grasp_contact_parameters(raw_model):
    for mesh in JAW_GEOMS:
        geom = _geom_by_mesh(raw_model, mesh)
        np.testing.assert_allclose(raw_model.geom_friction[geom], JAW_FRICTION)
        np.testing.assert_allclose(raw_model.geom_solref[geom], JAW_SOLREF)
        np.testing.assert_allclose(raw_model.geom_solimp[geom][:3], JAW_SOLIMP)


def test_reach_target_is_noncolliding_mocap_site(raw_model, sim_cfg):
    body = raw_model.body(sim_cfg.target_mocap_body)
    assert body.mocapid[0] >= 0
    assert raw_model.site(sim_cfg.target_site).bodyid[0] == body.id


def test_mocap_write_moves_target_site(backend, rng, sim_cfg):
    backend.reset(rng)
    xyz = np.array([0.05, 0.08, 0.62])
    backend.set_target_marker(xyz)
    site = backend.model.site(sim_cfg.target_site).id
    np.testing.assert_allclose(backend.data.site_xpos[site], xyz, atol=1e-12)


def test_uncompensated_arm_sags_down_settles_and_avoids_table(backend, rng, sim_cfg):
    start_z = backend.reset(rng).tcp_pos[2]
    for _ in range(ReachTaskConfig().max_episode_steps):
        backend.apply_delta(np.zeros(N_ARM))
        state = backend.step()
    drop = start_z - state.tcp_pos[2]
    assert drop > 0.0
    assert drop < MAX_SAG_M
    assert np.max(np.abs(state.qvel[:N_ARM])) < SETTLED_JOINT_SPEED_RAD_S
    assert not _arm_touches_table(backend.model, backend.data, sim_cfg)


def _arm_touches_table(model: mujoco.MjModel, data: mujoco.MjData, cfg: SimConfig) -> bool:
    robot = _robot_bodies(model, cfg.robot_root_body)
    table = model.body(cfg.table_body).id
    for contact in data.contact[: data.ncon]:
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if table in bodies and bodies & robot:
            return True
    return False

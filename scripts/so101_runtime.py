"""Runtime experiments; the baseline MJCF and original scripts stay unchanged."""
from pathlib import Path
import math
import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/so101/so101.xml"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS = ARM_JOINTS + ["gripper"]
INITIAL_DEG = [0, -20, 40, 15, 0, 40]


def load_model(compensate=False, cube_collision=False):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    if compensate:
        root = model.body("robot_mount").id
        for body in range(1, model.nbody):
            ancestor = body
            while ancestor and ancestor != root:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == root:
                model.body_gravcomp[body] = 1.0
        # Required when changing an initially uncompensated compiled model.
        if hasattr(model, "flg_gravcomp"):
            model.flg_gravcomp = True
        else:
            raise RuntimeError("Runtime compensation requires MuJoCo with flg_gravcomp support.")
    if cube_collision:
        geom = model.geom("cube_geom").id
        model.geom_contype[geom] = 1
        model.geom_conaffinity[geom] = 1
        # Broad-phase collision filtering uses cached body masks too.
        body = int(model.geom_bodyid[geom])
        model.body_contype[body] = 1
        model.body_conaffinity[body] = 1
    return model


class Robot:
    def __init__(self, model):
        self.model = model
        self.qpos = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINTS])
        self.dofs = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINTS])
        self.actuators = np.array([model.actuator(n + "_motor").id for n in JOINTS])
        self.joints = np.array([model.joint(n).id for n in ARM_JOINTS])
        self.sites = [model.site(n).id for n in ("fixed_tip_site", "moving_tip_site")]

    def initial_data(self):
        data = mujoco.MjData(self.model)
        data.qpos[self.qpos] = np.deg2rad(INITIAL_DEG)
        data.ctrl[self.actuators] = data.qpos[self.qpos]
        mujoco.mj_forward(self.model, data)
        return data

    def center(self, data):
        return data.site_xpos[self.sites].mean(axis=0).copy()

    def solve(self, data, target, tolerance=0.001):
        # Work on a separate kinematic state; never teleport the live robot.
        state = mujoco.MjData(self.model)
        state.qpos[:] = data.qpos
        for iteration in range(500):
            mujoco.mj_forward(self.model, state)
            error = target - self.center(state)
            if np.linalg.norm(error) < tolerance:
                return state.qpos[self.qpos].copy(), float(np.linalg.norm(error)), iteration
            jac = np.zeros((3, self.model.nv))
            for site in self.sites:
                jp = np.zeros_like(jac)
                mujoco.mj_jacSite(self.model, state, jp, None, site)
                jac += jp / 2
            jac = jac[:, self.dofs[:5]]
            dq = 0.5 * jac.T @ np.linalg.solve(jac @ jac.T + 0.05**2 * np.eye(3), error)
            limits = self.model.jnt_range[self.joints]
            state.qpos[self.qpos[:5]] = np.clip(
                state.qpos[self.qpos[:5]] + np.clip(dq, -math.radians(5), math.radians(5)),
                limits[:, 0], limits[:, 1])
        raise RuntimeError("IK failed to reach tolerance after 500 iterations")

    def advance(self, data, seconds):
        for _ in range(round(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, data)
            if not np.all(np.isfinite(data.qpos)) or np.any(data.warning.number):
                raise RuntimeError("Nonfinite state or MuJoCo warning during simulation")
        # mj_step leaves kinematics at the pre-integration state.
        mujoco.mj_forward(self.model, data)

    def report(self, data, target):
        return {
            "tracking_error_mm": float(np.linalg.norm(target - self.center(data)) * 1000),
            "joint_error_deg": dict(zip(JOINTS, np.rad2deg(
                data.ctrl[self.actuators] - data.qpos[self.qpos]).tolist())),
            "max_joint_speed_rad_s": float(np.max(np.abs(data.qvel[self.dofs]))),
            "contacts": int(data.ncon),
            "warning_counts": data.warning.number.tolist(),
            "cube_position": data.xpos[self.model.body("cube").id].tolist(),
        }

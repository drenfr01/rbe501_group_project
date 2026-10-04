"""Shared SO-101 helpers; run the test scripts rather than this module.

Cartesian positions are world coordinates in meters. MuJoCo joint positions
and actuator commands are radians; *_DEG constants and reported errors use
degrees. Model geometry, limits, gains, and collision masks live in the MJCF.
"""
from pathlib import Path
import math
import time
import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/so101/so101.xml"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS = ARM_JOINTS + ["gripper"]
INITIAL_DEG = [0, -20, 40, 15, 0, 40]
ACTUATOR_TARGETS_DEG = [20, -20, 40, 15, 30, 10]


def load_model(compensate=None):
    """Load the MJCF; compensate=None preserves its robot-only compensation.

    True/False override robot subtree compensation in memory for comparison.
    Cube/environment compensation, gravity, geometry, and collisions are left
    as defined by the model. No XML is written.
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    if compensate is not None:
        root = model.body("robot_mount").id
        for body in range(1, model.nbody):
            ancestor = body
            while ancestor and ancestor != root:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == root:
                model.body_gravcomp[body] = float(compensate)
        # MuJoCo 3.14 caches whether any body needs gravity compensation.
        if hasattr(model, "flg_gravcomp"):
            model.flg_gravcomp = bool(np.any(model.body_gravcomp))
    return model


class Robot:
    """Named joint/actuator indices plus single-site position IK and reports.

    The first five joints belong to the arm; the sixth is the moving gripper.
    gripper_site is rigidly attached to the stationary gripper body.
    """
    def __init__(self, model):
        self.model = model
        self.qpos = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINTS])
        self.dofs = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINTS])
        self.actuators = np.array([model.actuator(n + "_motor").id for n in JOINTS])
        self.joints = np.array([model.joint(n).id for n in ARM_JOINTS])
        self.site_id = model.site("gripper_site").id

    def initial_data(self):
        """Create the INITIAL_DEG pose and matching actuator holds; cube uses MJCF pose."""
        data = mujoco.MjData(self.model)
        data.qpos[self.qpos] = np.deg2rad(INITIAL_DEG)
        data.ctrl[self.actuators] = data.qpos[self.qpos]
        mujoco.mj_forward(self.model, data)
        return data

    def tcp_position(self, data):
        """Return current gripper_site world XYZ in meters (after forward/advance)."""
        return data.site_xpos[self.site_id].copy()

    def solve(self, data, target, tolerance=0.001):
        """Solve world XYZ (meters) using position-only damped least squares.

        Uses a separate kinematic state, so the live robot is never teleported.
        Updates only five arm joints and clips to their limits; orientation is
        free and gripper opening stays fixed. Returns six commanded angles in
        radians, pure position error in meters, and iteration count. Tolerance
        defaults to 1 mm; failure to converge raises RuntimeError. This does not
        plan collision-free motion or advance physics.
        """
        state = mujoco.MjData(self.model)
        state.qpos[:] = data.qpos
        jac = np.zeros((3, self.model.nv))
        limits = self.model.jnt_range[self.joints]
        for iteration in range(501):
            mujoco.mj_forward(self.model, state)
            error = target - self.tcp_position(state)
            if np.linalg.norm(error) < tolerance:
                # Include the unchanged gripper angle in the actuator command.
                return state.qpos[self.qpos].copy(), float(np.linalg.norm(error)), iteration
            if iteration == 500:
                break
            mujoco.mj_jacSite(self.model, state, jac, None, self.site_id)
            arm_jac = jac[:, self.dofs[:5]]
            # Damping bounds joint updates near singular configurations.
            dq = 0.5 * arm_jac.T @ np.linalg.solve(
                arm_jac @ arm_jac.T + 0.05**2 * np.eye(3), error)
            state.qpos[self.qpos[:5]] = np.clip(
                state.qpos[self.qpos[:5]] + np.clip(dq, -math.radians(5), math.radians(5)),
                limits[:, 0], limits[:, 1])
        raise RuntimeError("IK failed to reach tolerance after 500 iterations")

    def advance(self, data, seconds):
        """Advance simulated seconds under existing controls, then refresh kinematics.

        Raises on nonfinite positions or MuJoCo warnings; collisions stay enabled.
        """
        for _ in range(round(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, data)
            if not np.all(np.isfinite(data.qpos)) or np.any(data.warning.number):
                raise RuntimeError("Nonfinite state or MuJoCo warning during simulation")
        # Refresh site positions after the final integration step.
        mujoco.mj_forward(self.model, data)

    def report(self, data, target=None):
        """Report positions (m), target-minus-actual joint errors (deg), speed (rad/s),
        settled contact pairs, and warnings. Optional world XYZ adds TCP error
        in mm. Only cube/table pairs are classified as expected in this baseline.
        """
        pairs = set()
        for contact in data.contact:
            pairs.add(tuple(sorted(self.model.body(int(self.model.geom_bodyid[g])).name
                                   for g in (contact.geom1, contact.geom2))))
        result = {
            "tcp_position": self.tcp_position(data).tolist(),
            "joint_error_deg": dict(zip(JOINTS, np.rad2deg(
                data.ctrl[self.actuators] - data.qpos[self.qpos]).tolist())),
            "max_joint_speed_rad_s": float(np.max(np.abs(data.qvel[self.dofs]))),
            "contacts": int(data.ncon), "contact_body_pairs": sorted(pairs),
            "unexpected_contact_pairs": sorted(p for p in pairs if p != ("cube", "table_frame")),
            "warning_counts": data.warning.number.tolist(),
            "cube_position": data.xpos[self.model.body("cube").id].tolist(),
        }
        if target is not None:
            result["tracking_error_mm"] = float(np.linalg.norm(target - self.tcp_position(data)) * 1000)
        return result


def ik_target(model, data):
    """Return world XYZ (m), 6 cm above the cube center in the supplied state.

    Tests call this before physics, fixing the target at its initial value.
    """
    return data.xpos[model.body("cube").id].copy() + [0, 0, 0.06]


def show_viewer(robot, data, target=None):
    """Continue physics with current controls until the passive viewer is closed.

    Optional world target (m) appears as a magenta sphere. Camera defaults suit
    the table scene; timestep sleeping provides approximate real-time playback.
    """
    import mujoco.viewer
    with mujoco.viewer.launch_passive(robot.model, data) as viewer:
        viewer.cam.lookat[:] = [0.02, 0, 0.5]
        viewer.cam.distance = 1.05
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -22
        while viewer.is_running():
            robot.advance(data, robot.model.opt.timestep)
            if target is not None:
                # Magenta marks the target; the model defines the TCP frame.
                viewer.user_scn.ngeom = 1
                mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                                  np.full(3, 0.007), target, np.eye(3).ravel(),
                                  np.array([1., 0., 1., 1.]))
            viewer.sync()
            time.sleep(robot.model.opt.timestep)

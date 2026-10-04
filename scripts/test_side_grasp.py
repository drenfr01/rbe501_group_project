"""Side grasp experiment. No MJCF edits or actuator gain changes.
Run headless by default; --viewer replays the same sequence with grasp arrows.
The jaw-axis task has rank two; rotation about that axis remains unconstrained.
"""
import argparse
import json
import time
import mujoco
import numpy as np
from so101_runtime import Robot, load_model

AXIS = np.array([0., 1., 0.])
ORIENTATION_WEIGHT = 0.005  # metres per unit axis error; position has unit weight
OPEN_DEG = 60.


def jaw(robot, data):
    vector = data.site_xpos[robot.sites[1]] - data.site_xpos[robot.sites[0]]
    return vector / np.linalg.norm(vector)


def solve_pose(robot, data, target, opening=None, iterations=700):
    model = robot.model
    state = mujoco.MjData(model)
    state.qpos[:] = data.qpos
    if opening is not None:
        state.qpos[robot.qpos[-1]] = opening
    best = None
    for iteration in range(iterations):
        mujoco.mj_forward(model, state)
        axis = jaw(robot, state)
        ep = target - robot.center(state)
        eo = AXIS - axis
        score = float(ep @ ep + ORIENTATION_WEIGHT**2 * (eo @ eo))
        if best is None or score < best[0]:
            best = (score, state.qpos[robot.qpos].copy())
        if np.linalg.norm(ep) < 0.0001 and np.linalg.norm(eo) < 0.001:
            break
        j1 = np.zeros((3, model.nv)); j2 = np.zeros_like(j1)
        jr = np.zeros_like(j1)
        mujoco.mj_jacSite(model, state, j1, jr, robot.sites[0])
        mujoco.mj_jacSite(model, state, j2, None, robot.sites[1])
        # With jaw opening held fixed, the arm rotates both tips rigidly.
        # du/dq = omega x u, from mj_jacSite's rotational Jacobian.
        ju = np.cross(jr.T, axis).T
        jac = np.vstack([((j1 + j2) / 2)[:, robot.dofs[:5]],
                         ORIENTATION_WEIGHT * ju[:, robot.dofs[:5]]])
        error = np.r_[ep, ORIENTATION_WEIGHT * eo]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.003**2 * np.eye(6), error)
        state.qpos[robot.qpos[:5]] = np.clip(
            state.qpos[robot.qpos[:5]] + 0.5 * np.clip(dq, -0.1, 0.1),
            model.jnt_range[robot.joints, 0], model.jnt_range[robot.joints, 1])
    state.qpos[robot.qpos] = best[1]
    state.ctrl[robot.actuators] = best[1]
    mujoco.mj_forward(model, state)
    return best[1], state


def finger_info(robot, data):
    model = robot.model
    cube_geom = model.geom("cube_geom").id
    cube_center = data.geom_xpos[cube_geom]
    rotation = data.geom_xmat[cube_geom].reshape(3, 3)
    half = model.geom_size[cube_geom]
    result = {}
    for name, body_name, site in zip(["fixed", "moving"],
            ["gripper_link", "moving_jaw_so101_v1_link"], robot.sites):
        body = model.body(body_name).id
        geoms = np.flatnonzero(model.geom_bodyid == body)
        distances = [float(mujoco.mj_geomDistance(model, data, int(g), cube_geom, 1., None))
                     for g in geoms]
        contacts = [c for c in data.contact if cube_geom in (c.geom1, c.geom2)
                    and int(model.geom_bodyid[int(c.geom2 if c.geom1 == cube_geom else c.geom1)]) == body]
        local = rotation.T @ (data.site_xpos[site] - cube_center)
        outside = np.maximum(np.abs(local) - half, 0)
        point_distance = np.linalg.norm(outside) + min(float(np.max(np.abs(local) - half)), 0.)
        result[name] = {"tip_world": data.site_xpos[site].tolist(),
                        "tip_cube_local": local.tolist(),
                        "tip_surface_distance_mm": float(point_distance * 1000),
                        "mesh_cube_distance_mm": min(distances) * 1000,
                        "cube_contact_count": len(contacts),
                        "contact_positions_cube_local": [(rotation.T @ (c.pos - cube_center)).tolist() for c in contacts]}
    f = result["fixed"]["tip_cube_local"]; g = result["moving"]["tip_cube_local"]
    result["opposite_Y_sides"] = bool(f[1] < -half[1] and g[1] > half[1])
    return result


def report(robot, data, target):
    u = jaw(robot, data)
    cube_center = data.xpos[robot.model.body("cube").id]
    return {**robot.report(data, target), "desired_center": target.tolist(),
            "center_to_cube_error_mm": float(np.linalg.norm(robot.center(data) - cube_center) * 1000),
            "jaw_axis_world": u.tolist(), "desired_jaw_axis": AXIS.tolist(),
            "orientation_error_deg": float(np.rad2deg(np.arccos(np.clip(u @ AXIS, -1, 1)))),
            "jaw_gap_mm": float(np.linalg.norm(data.site_xpos[robot.sites[1]] - data.site_xpos[robot.sites[0]]) * 1000),
            "fingers": finger_info(robot, data)}


def draw(viewer, robot, data, target):
    viewer.user_scn.ngeom = 0
    for pos, color in [(robot.center(data), [0, .4, 1, 1]), (target, [1, 0, 1, 1])]:
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                          np.full(3, .005), pos, np.eye(3).ravel(), np.array(color))
        viewer.user_scn.ngeom += 1
    for pos, axis, color in [(robot.center(data), jaw(robot, data), [0, .4, 1, 1]),
                             (target, AXIS, [1, 0, 1, 1])]:
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                          np.zeros(3), np.eye(3).ravel(), np.array(color))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .002, pos, pos + .07 * axis)
        viewer.user_scn.ngeom += 1
    viewer.sync()


def run(viewer_enabled=False):
    model = load_model(compensate=True, cube_collision=True)
    robot = Robot(model); data = robot.initial_data()
    robot.advance(data, 1.)
    cube_id = model.body("cube").id
    cube = data.xpos[cube_id].copy()
    half = model.geom_size[model.geom("cube_geom").id].copy()
    target = cube.copy()
    # Validate the proposed final OPEN pose before running any approach.
    command, state = solve_pose(robot, data, target, np.deg2rad(OPEN_DEG))
    validation = report(robot, state, target)
    fingers = validation["fingers"]
    validation["pose_acceptable"] = bool(validation["tracking_error_mm"] < 1.
        and validation["orientation_error_deg"] < 12.
        and fingers["opposite_Y_sides"]
        and all(fingers[n]["mesh_cube_distance_mm"] > 0 for n in ["fixed", "moving"]))
    print(json.dumps({"final_open_pose_validation": validation}, indent=2), flush=True)
    # Test the sequence even if acceptance fails, and report the failure honestly.
    stages = []; viewer = None
    checks = {"approach_contact_steps": 0}
    phase = "initial"
    if viewer_enabled:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [.05, 0, .53]
        viewer.cam.distance = .7; viewer.cam.azimuth = 135; viewer.cam.elevation = -20
    def advance(seconds, desired):
        if viewer is None:
            for _ in range(round(seconds / model.opt.timestep)):
                robot.advance(data, model.opt.timestep)
                if phase == "approach":
                    info = finger_info(robot, data)
                    checks["approach_contact_steps"] += int(any(info[n]["cube_contact_count"] for n in ["fixed", "moving"]))
        else:
            for _ in range(round(seconds / model.opt.timestep)):
                robot.advance(data, model.opt.timestep)
                if viewer.is_running(): draw(viewer, robot, data, desired)
                time.sleep(model.opt.timestep)
    try:
        # Open while clear of the cube.
        data.ctrl[robot.actuators[-1]] = np.deg2rad(OPEN_DEG)
        advance(1., target)
        for name, desired in [("pre_grasp", cube + [0, 0, 2 * half[2] + .025]),
                              ("approach", target)]:
            phase = name
            start = robot.center(data)
            for fraction in np.linspace(.025, 1, 40):
                point = start + fraction * (desired - start)
                command, _ = solve_pose(robot, data, point, np.deg2rad(OPEN_DEG), 180)
                data.ctrl[robot.actuators] = command
                advance(.08, point)
            advance(1., desired)
            stages.append({"stage": name, **report(robot, data, desired)})
            print(json.dumps(stages[-1], indent=2), flush=True)
        phase = "close"
        for angle in np.linspace(OPEN_DEG, 0, 121):
            # Follow modest cube motion during closure instead of pushing past it.
            target = data.xpos[cube_id].copy()
            command, _ = solve_pose(robot, data, target, np.deg2rad(angle), 100)
            data.ctrl[robot.actuators] = command
            advance(.05, target)
            info = finger_info(robot, data)
            if all(info[n]["cube_contact_count"] > 0 for n in ["fixed", "moving"]):
                break
        hold_opening = float(data.ctrl[robot.actuators[-1]])
        advance(1., target)
        stages.append({"stage": "close", **report(robot, data, target)})
        phase = "lift"
        start_height = float(data.xpos[cube_id, 2]); start = robot.center(data)
        lift_target = start + [0, 0, .04]
        for fraction in np.linspace(.025, 1, 40):
            point = start + fraction * (lift_target - start)
            # Use the actual obstructed jaw opening, preserving its contact width.
            command, _ = solve_pose(robot, data, point, None, 150)
            command[-1] = hold_opening
            data.ctrl[robot.actuators] = command
            advance(.08, point)
        advance(1., lift_target)
        final = report(robot, data, lift_target)
        lift_mm = float((data.xpos[cube_id, 2] - start_height) * 1000)
        stages.append({"stage": "lift", **final, "cube_lift_mm": lift_mm})
        verified = lift_mm > 20 and all(final["fingers"][n]["cube_contact_count"] > 0 for n in ["fixed", "moving"])
        result = {"chosen_axis": "+Y", "cube_dimensions_mm": (2 * half * 1000).tolist(),
                  "desired_grasp_center": cube.tolist(), "last_close_target": target.tolist(), "validation": validation,
                  "stages": stages, "grasp_verified": bool(verified), "cube_lift_mm": lift_mm,
                  "held_gripper_command_deg": float(np.rad2deg(hold_opening)), **checks}
        print(json.dumps(result, indent=2))
        return result
    finally:
        if viewer is not None: viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    run(parser.parse_args().viewer)

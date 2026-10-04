"""Headless pre-grasp, descend, close, and lift probe with runtime cube collisions.

Position-only IK does not guarantee a usable finger orientation. Report contact
and lift evidence rather than assuming that closing means a successful grasp.
"""
import json
import numpy as np
from so101_runtime import Robot, load_model


def cube_contacts(model, data):
    cube = model.geom("cube_geom").id
    pairs = set()
    for contact in data.contact:
        if cube in (contact.geom1, contact.geom2):
            other = int(contact.geom2 if contact.geom1 == cube else contact.geom1)
            pairs.add(model.body(int(model.geom_bodyid[other])).name)
    return sorted(pairs)


def run():
    model = load_model(compensate=True, cube_collision=True)
    robot = Robot(model)
    data = robot.initial_data()
    robot.advance(data, 1.0)
    cube_id = model.body("cube").id
    cube = data.xpos[cube_id].copy()
    table = model.geom("table_top").id
    expected_z = model.geom_pos[table, 2] + model.geom_size[table, 2] + 0.025
    if abs(cube[2] - expected_z) > 0.001:
        raise RuntimeError("Cube did not settle on the table")
    stages = []
    for name, height in [("pre_grasp", 0.06), ("descend", 0.0)]:
        target = cube + [0, 0, height]
        start = robot.center(data)
        # Cartesian waypoints reduce sudden actuator target changes.
        for fraction in np.linspace(0.05, 1.0, 20):
            command, error, _ = robot.solve(data, start + fraction * (target - start))
            data.ctrl[robot.actuators] = command
            robot.advance(data, 0.1)
        robot.advance(data, 2.0)
        stages.append({"stage": name, **robot.report(data, target),
                       "cube_contact_bodies": cube_contacts(model, data)})
    encountered = set()
    for angle in np.linspace(40, 0, 41):
        data.ctrl[robot.actuators[-1]] = np.deg2rad(angle)
        robot.advance(data, 0.05)
        encountered.update(cube_contacts(model, data))
    robot.advance(data, 1.0)
    finger_contacts = cube_contacts(model, data)
    stages.append({"stage": "close", **robot.report(data, target),
                   "cube_contact_bodies": finger_contacts,
                   "cube_contacts_seen_during_close": sorted(encountered)})
    before_lift = float(data.xpos[cube_id, 2])
    start = robot.center(data)
    lift_target = start + [0, 0, 0.04]
    for fraction in np.linspace(0.05, 1.0, 20):
        command, _, _ = robot.solve(data, start + fraction * (lift_target - start))
        data.ctrl[robot.actuators] = command
        robot.advance(data, 0.1)
    robot.advance(data, 1.0)
    lift = float(data.xpos[cube_id, 2] - before_lift)
    contacts = cube_contacts(model, data)
    both_fingers = {"gripper_link", "moving_jaw_so101_v1_link"}.issubset(contacts)
    stages.append({"stage": "lift_probe", **robot.report(data, lift_target),
                   "cube_contact_bodies": contacts, "cube_lift_mm": lift * 1000})
    return {"cube_collisions_enabled": True, "robot_gravity_compensation": True,
            "cube_gravity_compensation": float(model.body_gravcomp[cube_id]),
            "cube_settled_z": float(cube[2]), "stages": stages,
            "grasp_verified": bool(lift > 0.02 and both_fingers)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

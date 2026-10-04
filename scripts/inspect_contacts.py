"""HEADLESS contact inspection at the settled six-actuator test pose.

HOW TO RUN (from the project folder):
    uv run python scripts/inspect_contacts.py

WHAT YOU CAN CHANGE / OPTIONS:
    --help: show usage. No viewer or pose options are provided.
    The pose uses ACTUATOR_TARGETS_DEG in so101_runtime.py. For another pose,
    command it with test_all_actuators.py and inspect its contact-body report.
    Geometry and collision settings belong in models/so101/so101.xml.

BEHAVIOR / OUTPUT / SUCCESS:
    Runs physics for five simulated seconds, then prints each unique contact
    geom pair with geom IDs/names, owning bodies, and mesh names. Several
    contact points can belong to one geom pair, so the total contact count may
    exceed the number of pairs printed. Negative distance indicates contact
    penetration (reported in mm). Unnamed geoms are identified by ID/body/mesh.
    This helps identify unintended robot self-collisions and robot/environment
    collisions. It reports SETTLED contacts, not the entire motion history.
    Default success is cube_geom touching table_top, no robot contacts, and no
    MuJoCo warnings. Expected cube/table contact is not a fault.

RELATED FILES: test_all_actuators.py provides direct joint commands;
so101_runtime.py shares the model loading, pose defaults, and reporting.
"""
import argparse
import mujoco
import numpy as np
from so101_runtime import Robot, load_model, ACTUATOR_TARGETS_DEG


def geom_info(model, geom):
    mesh = None
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.mesh(int(model.geom_dataid[geom])).name
    return {"geom_id": int(geom), "geom_name": model.geom(int(geom)).name,
            "body": model.body(int(model.geom_bodyid[geom])).name, "mesh": mesh}


def run():
    robot = Robot(load_model())
    data = robot.initial_data()
    data.ctrl[robot.actuators] = np.deg2rad(ACTUATOR_TARGETS_DEG)
    robot.advance(data, 5.)
    seen = set()
    print(f"Settled contacts: {data.ncon}")
    for contact in data.contact:
        pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
        if pair not in seen:
            seen.add(pair)
            print(geom_info(robot.model, pair[0]))
            print(geom_info(robot.model, pair[1]))
            print(f"Contact distance: {contact.dist * 1000:.3f} mm")
    return robot.report(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    run()

"""POSITION IK test: command the documented gripper_site TCP to an XYZ position.

HOW TO RUN (from the project folder):
    uv run python scripts/test_ik.py
    uv run python scripts/test_ik.py --x 0.15 --y 0.00 --z 0.55
    uv run python scripts/test_ik.py --x 0.15 --y 0.00 --z 0.55 --headless

WHAT YOU CAN CHANGE / OPTIONS:
    --x, --y, --z: supply all three WORLD coordinates in METERS, or none.
    Without XYZ, the target is 6 cm above the cube's INITIAL center.
    --headless: print results and exit without a viewer. --help: show usage.
    XYZ values are positions, not joint angles. For joint-angle commands, use
    test_all_actuators.py. Gripper opening remains 40 degrees during this test.

BEHAVIOR / OUTPUT / SUCCESS:
    Position-only damped least-squares IK solves five arm joints within limits;
    orientation is unconstrained. The live robot then tracks the solution for
    five simulated seconds using the model's robot-only gravity compensation.
    JSON reports the target, pure IK error, settled tracking error in mm, joint
    errors in degrees, contacts, and warnings. The default case should have
    roughly 0.85 mm IK and tracking error, no warnings, and only cube/table
    contacts. Arbitrary XYZ targets may be unreachable or blocked by collision;
    this is an IK test, not collision-free motion planning.
    Without --headless, the viewer opens first; arm tracking runs for five
    simulated seconds (magenta sphere marks the target), then JSON prints.

RELATED FILES: so101_runtime.py contains the solver; models/so101/so101.xml
contains gripper_site, joint limits, actuators, and scene geometry.
"""
import argparse
import json
import numpy as np
from so101_runtime import Robot, load_model, ik_target, run_settling


def run(headless=False, target=None):
    robot = Robot(load_model())
    data = robot.initial_data()
    target = ik_target(robot.model, data) if target is None else np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Target must contain three finite world coordinates in meters")
    command, error, iterations = robot.solve(data, target)
    data.ctrl[robot.actuators] = command

    def build_result():
        return {"tcp_site": "gripper_site", "target": target.tolist(),
                "pure_ik_error_mm": error * 1000, "ik_iterations": iterations,
                **robot.report(data, target)}

    def emit():
        print(json.dumps(build_result(), indent=2))

    run_settling(robot, data, 5., headless=headless, on_complete=emit, viewer_target=target)
    return build_result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    for axis in ("x", "y", "z"):
        parser.add_argument("--" + axis, type=float, help=f"World {axis.upper()} coordinate in meters; supply all XYZ or none")
    args = parser.parse_args()
    xyz = (args.x, args.y, args.z)
    supplied = [value is not None for value in xyz]
    if any(supplied) and not all(supplied):
        parser.error("Supply all of --x, --y, and --z, or none for the default target")
    if all(supplied) and not np.all(np.isfinite(xyz)):
        parser.error("XYZ coordinates must be finite numbers in meters")
    run(args.headless, xyz if all(supplied) else None)

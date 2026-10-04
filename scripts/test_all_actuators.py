"""DIRECT JOINT COMMAND test for all six SO-101 position actuators.

HOW TO RUN (from the project folder):
    uv run python scripts/test_all_actuators.py
    uv run python scripts/test_all_actuators.py --shoulder-pan 20
    uv run python scripts/test_all_actuators.py --shoulder-pan 20 --shoulder-lift -15 --elbow-flex 40
    uv run python scripts/test_all_actuators.py --gripper 30

WHAT YOU CAN CHANGE / OPTIONS:
    --shoulder-pan, --shoulder-lift, --elbow-flex, --wrist-flex, --wrist-roll,
    --gripper: target joint angles in DEGREES. Omitted joints retain defaults:
    [20, -20, 40, 15, 30, 10] in the above order. Current limits are read directly
    from the MJCF and shown by --help; out-of-range commands are rejected.
    --headless: run without a viewer. --help: show usage and joint limits.
    These are JOINT commands. Use test_ik.py for Cartesian XYZ positions.

BEHAVIOR / OUTPUT / SUCCESS:
    Commands are converted to radians and tracked for five simulated seconds
    with robot-only gravity compensation. JSON prints commanded targets,
    resulting joint positions, target-minus-actual errors (all in degrees),
    contacts, and warnings. Defaults should settle near the commanded angles
    with no warnings and only expected cube/table contact. Joint limits do not
    guarantee collision-free poses. Without --headless, the viewer opens first
    and joint motion is visible during the five-second settle; JSON prints when
    that interval finishes and the window stays open until you close it.

RELATED FILES: defaults and simulation helpers are in so101_runtime.py;
limits, actuator gains, geometry, and collision settings are in so101.xml.
"""
import argparse
import json
import numpy as np
from so101_runtime import Robot, load_model, run_settling, ACTUATOR_TARGETS_DEG, JOINTS


def run(headless=False, targets_deg=None, model=None):
    robot = Robot(load_model() if model is None else model)
    data = robot.initial_data()
    targets = np.array(ACTUATOR_TARGETS_DEG if targets_deg is None else targets_deg, dtype=float)
    limits = np.rad2deg(robot.model.jnt_range[[robot.model.joint(n).id for n in JOINTS]])
    if targets.shape != (6,) or not np.all(np.isfinite(targets)):
        raise ValueError("Supply six finite joint targets in degrees")
    for name, target, (lower, upper) in zip(JOINTS, targets, limits):
        if not lower <= target <= upper:
            raise ValueError(f"{name}: {target:g} deg is outside [{lower:.6f}, {upper:.6f}] deg")
    data.ctrl[robot.actuators] = np.deg2rad(targets)
    print("Commanded joint targets (degrees):")
    print(json.dumps(dict(zip(JOINTS, targets.tolist())), indent=2))

    def build_result():
        return {"commanded_joint_targets_deg": dict(zip(JOINTS, targets.tolist())),
                "actual_joint_positions_deg": dict(zip(JOINTS, np.rad2deg(data.qpos[robot.qpos]).tolist())),
                **robot.report(data)}

    def emit():
        print(json.dumps(build_result(), indent=2))

    run_settling(robot, data, 5., headless=headless, on_complete=emit)
    return build_result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    model = load_model()
    limits = np.rad2deg(model.jnt_range[[model.joint(n).id for n in JOINTS]])
    for name, default, (lower, upper) in zip(JOINTS, ACTUATOR_TARGETS_DEG, limits):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default,
                            help=f"Degrees; default {default:g}; MJCF limits [{lower:.6f}, {upper:.6f}]")
    args = parser.parse_args()
    targets = [getattr(args, name) for name in JOINTS]
    for name, target, (lower, upper) in zip(JOINTS, targets, limits):
        if not np.isfinite(target) or not lower <= target <= upper:
            parser.error(f"--{name.replace('_', '-')}: use a finite angle in [{lower:.6f}, {upper:.6f}] degrees")
    run(args.headless, targets, model)

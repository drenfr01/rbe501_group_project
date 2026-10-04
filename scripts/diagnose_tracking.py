"""Compare identical IK commands with gravity off, on, and compensated."""
import argparse
import json
import numpy as np
from so101_runtime import Robot, load_model


def run(seconds=5.0):
    baseline = load_model()
    robot = Robot(baseline)
    initial = robot.initial_data()
    target = initial.xpos[baseline.body("cube").id].copy() + [0, 0, 0.06]
    solution, pure_error, iterations = robot.solve(initial, target)
    results = {"mujoco_version": __import__("mujoco").__version__,
               "duration_s": seconds, "target": target.tolist(),
               "pure_ik_error_mm": pure_error * 1000, "ik_iterations": iterations,
               "cube_collisions_enabled": False, "cases": {}}
    for name, gravity, compensate in [
        ("gravity_off", False, False), ("gravity_on", True, False),
        ("gravity_on_compensated", True, True)]:
        model = load_model(compensate=compensate)
        if not gravity:
            model.opt.gravity[:] = 0
        case_robot = Robot(model)
        data = case_robot.initial_data()
        data.ctrl[case_robot.actuators] = solution
        case_robot.advance(data, seconds)
        results["cases"][name] = case_robot.report(data, target)
    compensated = results["cases"]["gravity_on_compensated"]
    results["compensation_matches_ik_floor"] = bool(
        abs(compensated["tracking_error_mm"] - pure_error * 1000) < 0.1
        and compensated["max_joint_speed_rad_s"] < 0.001)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    results = run(args.seconds)
    print(json.dumps(results, indent=2))
    if not results["compensation_matches_ik_floor"]:
        raise SystemExit(1)

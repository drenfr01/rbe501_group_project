"""HEADLESS tracking diagnostic: separate geometric IK error from gravity sag.

HOW TO RUN (from the project folder):
    uv run python scripts/diagnose_tracking.py
    uv run python scripts/diagnose_tracking.py --seconds 5

WHAT YOU CAN CHANGE / OPTIONS:
    --seconds: positive settling duration in simulated seconds (default 5).
    --help: show usage. No viewer is opened. The shared ik_target helper sets
    the target 6 cm above the initial cube center; keep it fixed across cases.

BEHAVIOR / OUTPUT:
    Solves position IK once for gripper_site and reports its pure kinematic
    error. Three independent simulations receive the SAME joint commands:
      1. gravity off, compensation off;
      2. gravity on, compensation off;
      3. gravity on, robot-only compensation on.
    IK may be geometrically correct while position actuators sag under gravity.
    Compensation removes robot weight loading without removing cube gravity.
    Only in-memory model settings are overridden; the MJCF is not rewritten.
    JSON gives error in mm, joint errors in degrees, speeds, contacts, warnings,
    and compensation_matches_ik_floor. Cube collisions stay enabled; gravity
    off leaves the cube at its initial height, while gravity on settles it.

SUCCESS:
    Default pure IK/gravity-off/compensated errors are about 0.85 mm, versus
    about 11.6 mm uncompensated. Compensation should match the IK floor within
    0.1 mm with negligible joint speed and no unexpected contacts. The script
    exits nonzero if this comparison fails. Short durations may not settle.
    Expected cube/table contacts are normal; MuJoCo warnings are not.

RELATED FILES: test_ik.py runs a single compensated target; so101_runtime.py
implements IK, compensation overrides, and reports.
"""
import argparse
import json
import math
import mujoco
from so101_runtime import Robot, load_model, ik_target


def run(seconds=5.0):
    robot = Robot(load_model())
    initial = robot.initial_data()
    target = ik_target(robot.model, initial)
    solution, pure_error, iterations = robot.solve(initial, target)
    results = {"mujoco_version": mujoco.__version__, "tcp_site": "gripper_site",
               "duration_s": seconds, "target": target.tolist(),
               "pure_ik_error_mm": pure_error * 1000, "ik_iterations": iterations,
               "cube_collisions_enabled": True, "cases": {}}
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
        and compensated["max_joint_speed_rad_s"] < 0.001
        and not compensated["unexpected_contact_pairs"])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=5)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    results = run(args.seconds)
    print(json.dumps(results, indent=2))
    if not results["compensation_matches_ik_floor"]:
        raise SystemExit(1)

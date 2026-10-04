"""SCENE / VIEWER sanity check for the mounted SO-101 and its environment.

HOW TO RUN (from the project folder):
    uv run python scripts/test_table_cube.py
    uv run python scripts/test_table_cube.py --headless

WHAT YOU CAN CHANGE / OPTIONS:
    --headless: print the settled scene report and exit. --help: show usage.
    Edit cube, robot mounting, table, floor, and lighting placement in
    models/so101/so101.xml. The initial joint pose is INITIAL_DEG in
    so101_runtime.py. This script has no placement controls.

BEHAVIOR / OUTPUT / SUCCESS:
    Loads the robot, table, cube, floor, and lighting. Physics RUNS: the cube
    settles under gravity while compensated position actuators hold the robot
    in its initial pose. After two simulated seconds, JSON reports TCP/cube
    positions in meters, joint errors in degrees, contacts, and warnings.
    The viewer then continues physics; this is not a static rendering.
    Success means correct visual placement and mounting, a cube resting on
    the tabletop (center z about 0.500 m), no warnings, and only cube/table
    contacts. Use this before troubleshooting actuators or IK.

RELATED FILES: so101_runtime.py provides loading and viewer helpers;
inspect_contacts.py identifies individual contact geometries.
"""
import argparse
import json
from so101_runtime import Robot, load_model, run_settling


def run(headless=False):
    robot = Robot(load_model())
    data = robot.initial_data()

    def emit():
        print(json.dumps(robot.report(data), indent=2))

    run_settling(robot, data, 2., headless=headless, on_complete=emit)
    return robot.report(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    run(parser.parse_args().headless)

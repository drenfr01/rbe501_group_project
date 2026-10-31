"""Pose check: send the real arm to poses defined in sim degrees and measure the links.

Each pose in `PoseCheckConfig.poses` is a sim joint vector converted to
LeRobot degrees with `HardwareConfig.joint_map`. If the map is right, the
physical links sit at the angles the pose describes (upper arm vertical,
forearm level, ...). Hold a phone level app against the link and type what it
reads at the prompt.

Before torque is enabled the whole path (start, every pose, back to start) is
checked against the table, with a sag allowance; the run is refused if any
point comes too close. Moves are ramped at `ramp_speed_deg_s`. Keep the power
switch within reach; Ctrl-C ramps back to the start pose before torque off.

Results go to data/pose_check/<date>_<robot_id>/results.json. `--sim` runs the
same code against MuJoCo as a dry run.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from rbe_501_so101.config import CheckPose, HardwareConfig, PoseCheckConfig, SimConfig, resolve_project_path
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.operator import TerminalKeys
from rbe_501_so101.hardware.pose_check import (
    MM_PER_M,
    PoseCheckRepository,
    PoseCheckResult,
    PoseCheckRunner,
    plan_path,
    preflight,
)
from rbe_501_so101.hardware.safety import SafetyAbort, SafetyMonitor
from rbe_501_so101.hardware.session import open_bus, robot_id_for
from rbe_501_so101.provenance import begin_run, experiment_dir, require_clean_tree


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command line."""
    hw = HardwareConfig()
    names = [p.name for p in PoseCheckConfig().poses]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=hw.port)
    parser.add_argument("--robot-id", default=hw.robot_id)
    parser.add_argument("--poses", nargs="+", choices=names, default=names, help="subset of poses, in order")
    parser.add_argument("--out", type=Path, default=None, help="folder for results.json")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--sim", action="store_true", help="dry run on the MuJoCo arm")
    return parser.parse_args(argv)


def selected(names: list[str], cfg: PoseCheckConfig) -> tuple[CheckPose, ...]:
    """Configured poses in the order given on the command line."""
    by_name = {p.name: p for p in cfg.poses}
    return tuple(by_name[n] for n in names)


def check_path(joint_map: JointMap, model: mujoco.MjModel, start: dict[str, float], poses: tuple[CheckPose, ...], cfg: PoseCheckConfig) -> None:
    """Refuse the run (before torque on) if the planned path comes near the table."""
    q_start = joint_map.to_sim(start)
    q_poses = tuple(np.append(np.deg2rad(p.sim_deg), q_start[-1]) for p in poses)
    margin = preflight(SafetyMonitor.from_model(model, SimConfig()), plan_path(q_start, q_poses), cfg)
    print(f"path check passed: lowest point stays {MM_PER_M * margin:.0f} mm above the table clearance (with sag allowance)")


def show(result: PoseCheckResult, joints: tuple[str, ...]) -> None:
    """Goal vs reading per joint, plus what to measure."""
    print(f"\n== {result.name}{'   (WARNING: LeRobot clamped a goal)' if result.clamped else ''}")
    print(f"   {'joint':15s} {'sim goal':>9s} {'sim read':>9s} {'lr goal':>9s} {'lr read':>9s} {'error':>7s}")
    for i, sim_goal in enumerate(result.sim_goal_deg):
        goal, read = result.lerobot_goal_deg[i], result.measured_deg[i]
        print(f"   {joints[i]:15s} {sim_goal:9.2f} {result.measured_sim_deg[i]:9.2f} {goal:9.2f} {read:9.2f} {read - goal:7.2f}")
    print(f"   expect: {result.check}")


def run(runner: PoseCheckRunner, poses: tuple[CheckPose, ...], repo: PoseCheckRepository, sim: bool) -> None:
    """Visit every pose, ask for the level reading, and save after each pose."""
    results: list[PoseCheckResult] = []
    with TerminalKeys() as keys:
        for pose in poses:
            result = runner.visit(pose)
            show(result, SimConfig().joints)
            note = "sim dry run" if sim else keys.prompt("   measured (e.g. 'forearm -2, gripper +1'): ")
            results.append(result.with_note(note))
            repo.save(tuple(results))
    print(f"\nsaved {len(results)} poses to {repo.path}")


def main(argv: list[str] | None = None) -> None:
    """Entry point; the path check runs before torque is enabled."""
    args = parse_args(argv)
    require_clean_tree(args.allow_dirty)
    hw, cfg, sim_cfg = replace(HardwareConfig(), port=args.port, robot_id=args.robot_id), PoseCheckConfig(), SimConfig()
    poses, robot_id = selected(args.poses, cfg), robot_id_for(args.sim, hw)
    out = resolve_project_path(args.out) if args.out else experiment_dir(robot_id, base=cfg.results_dir)
    model = mujoco.MjModel.from_xml_path(str(sim_cfg.model_path))
    joint_map = JointMap.from_model(model, sim_cfg, hw)
    bus, clock = open_bus(args.sim, hw)
    try:
        check_path(joint_map, model, bus.read_positions(), poses, cfg)
    except SafetyAbort as exc:
        bus.close()
        sys.exit(f"refusing to move: {exc}")
    manifest = begin_run(out, robot_id)
    runner = PoseCheckRunner(bus, clock, joint_map, cfg, sim_cfg.joints)
    runner.start()
    try:
        run(runner, poses, PoseCheckRepository(out / cfg.results_filename), args.sim)
    finally:
        runner.finish()
        bus.close()
    manifest.finalize(out)


if __name__ == "__main__":
    main()

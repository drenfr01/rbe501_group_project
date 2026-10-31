"""Phase 0 real sag check for the SO-101 follower.

Subcommands:
  record-poses  torque off; move the arm by hand and press `r` to save the
                current pose to the poses file, `q` to finish.
  run           approach every saved pose from above and below, settle, and
                average readings. Writes data/sag_check/<date>_<robot_id>/.

Keep the power switch within reach. `--sim` runs `run` against MuJoCo
through the same bus interface (robot id "sim") as a dry run.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from rbe_501_so101.config import HardwareConfig, SagCheckConfig, SimConfig, resolve_project_path
from rbe_501_so101.hardware.bus import LeRobotJointBus
from rbe_501_so101.hardware.joint_mapping import pos_key
from rbe_501_so101.hardware.operator import ABORT_KEY, TerminalKeys, redraw
from rbe_501_so101.hardware.sag import RecordedPose, SagRepository, SagSession
from rbe_501_so101.hardware.sag_procedure import SagCheckRunner
from rbe_501_so101.hardware.session import open_bus, robot_id_for
from rbe_501_so101.provenance import begin_run, require_clean_tree

RECORD_KEY = "r"
POLL_S = 0.1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command line for both subcommands."""
    hw = HardwareConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("record-poses", "run"))
    parser.add_argument("--port", default=hw.port)
    parser.add_argument("--robot-id", default=hw.robot_id)
    parser.add_argument("--poses", type=Path, default=SagCheckConfig().poses_path, help="pose list JSON")
    parser.add_argument("--out", type=Path, default=None, help="session folder (default data/sag_check/<date>_<id>)")
    parser.add_argument("--allow-dirty", action="store_true", help="run even with uncommitted changes")
    parser.add_argument("--sim", action="store_true", help="dry-run `run` on the MuJoCo arm")
    return parser.parse_args(argv)


def joints() -> tuple[str, ...]:
    """Session joint order: arm joints, then the gripper."""
    sim = SimConfig()
    return (*sim.arm_joints, sim.gripper_joint)


def record_poses(repo: SagRepository, hw: HardwareConfig) -> None:
    """Torque off; save a pose per keypress until the operator quits."""
    bus = LeRobotJointBus.connect(hw)
    bus.set_torque(False)
    poses: list[RecordedPose] = []
    try:
        with TerminalKeys() as keys:
            poses = _record_loop(bus, keys)
    finally:
        bus.close()
    print(f"saved {len(poses)} poses to {repo.save_poses(joints(), tuple(poses))}")


def _record_loop(bus: LeRobotJointBus, keys: TerminalKeys) -> list[RecordedPose]:
    """Show live readings; `r` appends the current pose, `q` stops."""
    poses: list[RecordedPose] = []
    first = True
    while (key := keys.poll(POLL_S)) != ABORT_KEY:
        reading = bus.read_positions()
        values = tuple(round(reading[pos_key(j)], 2) for j in joints())
        if key == RECORD_KEY:
            poses.append(RecordedPose(len(poses), values))
        lines = [f"poses saved: {len(poses)}   [{RECORD_KEY}] record  [{ABORT_KEY}] finish"]
        lines += [f"  {j:15s} {v:8.2f}" for j, v in zip(joints(), values, strict=True)]
        redraw(lines, first)
        first = False
    return poses


def report_clamps(session: SagSession, clamp_count: int) -> None:
    """Say which samples LeRobot clipped (they are excluded from the analysis)."""
    flagged = [f"pose {s.pose_id} {s.approach}" for s in session.samples if s.clamped]
    if not flagged:
        print("no goals were clamped by LeRobot")
        return
    print(f"WARNING: LeRobot clamped {clamp_count} goals; {len(flagged)} samples flagged and excluded from analysis:")
    print("  " + ", ".join(flagged))


def run_session(repo: SagRepository, args: argparse.Namespace, hw: HardwareConfig) -> None:
    """Run every pose from both sides, save the session, and ramp back to the start pose."""
    pose_joints, poses = repo.load_poses()
    if pose_joints != joints():
        sys.exit(f"pose file joints {pose_joints} do not match {joints()}")
    robot_id = robot_id_for(args.sim, hw)
    out = resolve_project_path(args.out) if args.out else repo.session_dir(robot_id)
    manifest = begin_run(out, robot_id)
    bus, clock = open_bus(args.sim, hw)
    runner = SagCheckRunner(bus, clock, SagCheckConfig(), joints())
    rest = bus.read_positions()
    try:
        session = runner.session(robot_id, poses)
        print(f"session saved to {repo.save_session(session, out)}")
        report_clamps(session, runner.clamp_count)
    finally:
        runner.finish(rest)
        bus.close()
    manifest.finalize(out)


def main(argv: list[str] | None = None) -> None:
    """Entry point; refuses a dirty tree before touching the arm."""
    args = parse_args(argv)
    require_clean_tree(args.allow_dirty)
    hw = replace(HardwareConfig(), port=args.port, robot_id=args.robot_id)
    repo = SagRepository(replace(SagCheckConfig(), poses_path=resolve_project_path(args.poses)))
    if args.command == "record-poses":
        if args.sim:
            sys.exit("record-poses needs the real arm (poses are jogged by hand)")
        record_poses(repo, hw)
        return
    run_session(repo, args, hw)


if __name__ == "__main__":
    main()

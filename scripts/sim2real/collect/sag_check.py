"""Phase 0 real sag check for the SO-101 follower.

Subcommands:
  record-poses  torque off; move the arm by hand and press `r` to save the
                current pose to the poses file, `q` to finish.
  run           approach every saved pose from above and below, settle, and
                average readings. Writes
                data/sag_check/<date>_<robot_id>_p<gain>_<HHMMSS>/.

"From above" is the side gravity pulls away from, per joint and pose, so the
approach point lifts the arm; every pose is entered through a raised waypoint
(`SagCheckConfig.waypoint_sim_deg`). Before torque is enabled the whole path
(waypoints, both approach points, every pose, back to rest) is checked
against the table with a sag allowance, and the run is refused if it comes
too close. Keep the power switch within reach. `--sim` runs `run` against MuJoCo
through the same bus interface (robot id "sim") as a dry run.
"""

import argparse
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import mujoco

from rbe_501_so101.config import HardwareConfig, SagCheckConfig, SimConfig, resolve_project_path
from rbe_501_so101.hardware.bus import JointBus, LeRobotJointBus
from rbe_501_so101.hardware.clock import Clock
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.operator import ABORT_KEY, TerminalKeys, redraw
from rbe_501_so101.hardware.path_safety import MM_PER_M, preflight
from rbe_501_so101.hardware.safety import SafetyAbort, SafetyMonitor
from rbe_501_so101.hardware.sag import RecordedPose, SagRepository, SagSession
from rbe_501_so101.hardware.sag_procedure import (
    GravityAwareOffsets,
    SagCheckRunner,
    lerobot_waypoint,
    with_approach_offset,
)
from rbe_501_so101.hardware.session import open_bus, robot_id_for
from rbe_501_so101.kinematics.gravity import GravityTorqueModel
from rbe_501_so101.provenance import begin_run, require_clean_tree, timestamped_suffix

RECORD_KEY = "r"
POLL_S = 0.1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command line for both subcommands."""
    hw = HardwareConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("record-poses", "run"))
    parser.add_argument("--port", default=hw.port)
    parser.add_argument("--robot-id", default=hw.robot_id)
    parser.add_argument("--p-gain", type=int, default=hw.position_p_gain, help="servo P_Coefficient written on connect")
    parser.add_argument(
        "--approach-offset",
        type=float,
        default=SagCheckConfig().approach_offset_deg,
        help="degrees above/below each pose to approach from (larger sweeps the whole friction band)",
    )
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
    """Say how often LeRobot clipped ramp goals, and which samples never got their final goal (excluded)."""
    flagged = [f"pose {s.pose_id} {s.approach}" for s in session.samples if s.clamped]
    print(f"LeRobot clipped {clamp_count} ramp goals (a joint lagged the ramp by more than its relative-target cap)")
    if flagged:
        print(f"WARNING: {len(flagged)} samples never got their final goal and are excluded from analysis:")
        print("  " + ", ".join(flagged))


def build_runner(bus: JointBus, clock: Clock, cfg: SagCheckConfig, joint_map: JointMap) -> SagCheckRunner:
    """Runner with gravity-aware approach sides and the raised waypoint (LeRobot units)."""
    offsets = GravityAwareOffsets(
        joint_map, GravityTorqueModel.from_config(SimConfig()), cfg.approach_offset_deg, cfg.gravity_torque_floor_nm
    )
    return SagCheckRunner(bus, clock, cfg, joints(), offsets, lerobot_waypoint(joint_map, cfg.waypoint_sim_deg))


def check_path(
    runner: SagCheckRunner,
    poses: tuple[RecordedPose, ...],
    rest: dict[str, float],
    joint_map: JointMap,
    model: mujoco.MjModel,
    cfg: SagCheckConfig,
) -> None:
    """Refuse the run (before torque on) if any planned point comes near the table."""
    path = [joint_map.to_sim(goal) for goal in [rest, *runner.plan(poses, rest)]]
    try:
        margin = preflight(SafetyMonitor.from_model(model, SimConfig()), path, cfg.path_check)
    except SafetyAbort as exc:
        sys.exit(f"refusing to move: {exc}. Re-record the poses higher.")
    print(f"path check passed: lowest point stays {MM_PER_M * margin:.0f} mm above the table clearance (with sag allowance)")


def session_dir(args: argparse.Namespace, repo: SagRepository, robot_id: str, hw: HardwareConfig) -> Path:
    """`--out`, or a new timestamped session folder per run."""
    if args.out:
        return resolve_project_path(args.out)
    return repo.session_dir(timestamped_suffix(robot_id, hw.position_p_gain, datetime.now()))


def run_session(repo: SagRepository, args: argparse.Namespace, hw: HardwareConfig, cfg: SagCheckConfig) -> None:
    """Check the path, run every pose from both sides, save the session, and ramp back to the start pose."""
    pose_joints, poses = repo.load_poses()
    if pose_joints != joints():
        sys.exit(f"pose file joints {pose_joints} do not match {joints()}")
    robot_id = robot_id_for(args.sim, hw)
    model = mujoco.MjModel.from_xml_path(str(SimConfig().model_path))
    joint_map = JointMap.from_model(model, SimConfig(), hw)
    bus, clock = open_bus(args.sim, hw)
    runner, rest = build_runner(bus, clock, cfg, joint_map), bus.read_positions()
    try:
        check_path(runner, poses, rest, joint_map, model, cfg)
    except SystemExit:
        bus.close()
        raise
    out = session_dir(args, repo, robot_id, hw)
    manifest = begin_run(out, robot_id)
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
    hw = replace(HardwareConfig(), port=args.port, robot_id=args.robot_id, position_p_gain=args.p_gain)
    try:
        base = replace(SagCheckConfig(), poses_path=resolve_project_path(args.poses))
        cfg = with_approach_offset(base, args.approach_offset)
    except ValueError as exc:
        sys.exit(str(exc))
    repo = SagRepository(cfg)
    if args.command == "record-poses":
        if args.sim:
            sys.exit("record-poses needs the real arm (poses are jogged by hand)")
        record_poses(repo, hw)
        return
    print(f"approach offset {cfg.approach_offset_deg:g} deg")
    run_session(repo, args, hw, cfg)


if __name__ == "__main__":
    main()

"""Replay hardware procedures (sag check, pose check) on the calibrated simulated arm.

The replay drives the same runners through a `SimJointBus`, so the sim sees the
same ramps, approach offsets, waypoint, LeRobot relative clip, and encoder
rounding as the real arm did.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from rbe_501_so101.config import (
    ENCODER_QUANTIZATION_RAD,
    CheckPose,
    DynamicsRandomizationConfig,
    HardwareConfig,
    PoseCheckConfig,
    SagCheckConfig,
    ServoEffectsConfig,
    SimConfig,
)
from rbe_501_so101.envs.factory import servo_effects
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.pose_check import PoseCheckResult, PoseCheckRunner
from rbe_501_so101.hardware.sag import RecordedPose, SagSession, usable_samples
from rbe_501_so101.hardware.sag_procedure import RecordedOffsets, SagCheckRunner, UniformOffsets, lerobot_waypoint
from rbe_501_so101.hardware.sim_bus import CommandedBackend, SimJointBus

SIM_ROBOT_ID = "sim"


@dataclass(frozen=True)
class ReplaySetup:
    """The simulated arm a replay runs on.

    `sim_cfg.calibration_path` sets stiffness and the stiction band; `servo`
    chooses which servo effects apply (LeRobot's clip is done by the bus, as
    on hardware). `start_sim_deg` is where the arm starts (the real runs start
    from rest); None keeps the sim home pose.
    """

    sim_cfg: SimConfig
    joint_map: JointMap
    servo: ServoEffectsConfig = ServoEffectsConfig()
    quantization_rad: float = ENCODER_QUANTIZATION_RAD
    max_relative_deg: float | None = HardwareConfig().max_relative_target_deg
    start_sim_deg: tuple[float, ...] | None = HardwareConfig().rest_deg


def open_sim_bus(setup: ReplaySetup) -> SimJointBus:
    """Unrandomized calibrated arm (optionally with servo effects) behind a LeRobot-unit bus."""
    sim_cfg = setup.sim_cfg if setup.start_sim_deg is None else replace(setup.sim_cfg, initial_deg=setup.start_sim_deg)
    backend: CommandedBackend = MujocoSo101Backend(sim_cfg, DynamicsRandomizationConfig(enabled=False))
    if setup.servo.active:
        backend = servo_effects(backend, setup.servo, sim_cfg)
    return SimJointBus(backend, setup.joint_map, sim_cfg, setup.quantization_rad, setup.max_relative_deg)


def replay_sag_session(
    real: SagSession, setup: ReplaySetup, cfg: SagCheckConfig, waypoint_sim_deg: tuple[float, ...] | None
) -> SagSession:
    """Run the sag procedure at the real goals, with the real approach offsets (and waypoint, if used)."""
    bus = open_sim_bus(setup)
    goals = {s.pose_id: s.goal_deg for s in usable_samples(real).samples}
    poses = tuple(RecordedPose(pid, goal) for pid, goal in goals.items())
    offsets = RecordedOffsets(real, UniformOffsets(cfg.approach_offset_deg, real.joints))
    waypoint = None if waypoint_sim_deg is None else lerobot_waypoint(setup.joint_map, waypoint_sim_deg)
    try:
        return SagCheckRunner(bus, bus, cfg, real.joints, offsets, waypoint).session(SIM_ROBOT_ID, poses)
    finally:
        bus.close()


def replay_pose_check(
    names: Sequence[str], setup: ReplaySetup, cfg: PoseCheckConfig, poses: Mapping[str, CheckPose]
) -> tuple[PoseCheckResult, ...]:
    """Visit the named check poses in order, as the real run did."""
    bus = open_sim_bus(setup)
    runner = PoseCheckRunner(bus, bus, setup.joint_map, cfg, setup.sim_cfg.joints)
    try:
        runner.start()
        return tuple(runner.visit(poses[name]) for name in names)
    finally:
        bus.close()

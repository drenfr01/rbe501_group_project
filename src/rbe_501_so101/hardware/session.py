"""Wiring shared by the hardware scripts: open the real or simulated bus and build the real backend."""

import mujoco

from rbe_501_so101.config import (
    ENCODER_QUANTIZATION_RAD,
    HardwareConfig,
    IKConfig,
    SafetyConfig,
    SimConfig,
)
from rbe_501_so101.envs.model_indices import ModelIndices
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend
from rbe_501_so101.hardware.bus import JointBus, LeRobotJointBus
from rbe_501_so101.hardware.clock import Clock, SystemClock
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.real_backend import LeRobotSo101Backend
from rbe_501_so101.hardware.safety import SafetyMonitor
from rbe_501_so101.hardware.sim_bus import SimJointBus
from rbe_501_so101.kinematics import PositionIKSolver

SIM_ROBOT_ID = "sim"


def open_bus(
    sim: bool,
    hw: HardwareConfig = HardwareConfig(),
    sim_cfg: SimConfig = SimConfig(),
    quantization_rad: float = ENCODER_QUANTIZATION_RAD,
) -> tuple[JointBus, Clock]:
    """Connect to the arm, or open a MuJoCo stand-in whose clock is simulated time."""
    if not sim:
        return LeRobotJointBus.connect(hw), SystemClock()
    backend = MujocoSo101Backend(sim_cfg)
    joint_map = JointMap.from_model(backend.model, sim_cfg, hw)
    bus = SimJointBus(backend, joint_map, sim_cfg, quantization_rad, hw.max_relative_target_deg)
    return bus, bus


def robot_id_for(sim: bool, hw: HardwareConfig) -> str:
    """Robot id recorded in manifests and session names."""
    return SIM_ROBOT_ID if sim else hw.robot_id


def build_real_backend(
    bus: JointBus,
    clock: Clock,
    hw: HardwareConfig = HardwareConfig(),
    sim_cfg: SimConfig = SimConfig(),
    safety_cfg: SafetyConfig = SafetyConfig(),
    home_deg: tuple[float, ...] | None = None,
) -> LeRobotSo101Backend:
    """Real backend over `bus` with the configured joint map, safety monitor, and kinematics model."""
    model = mujoco.MjModel.from_xml_path(str(sim_cfg.model_path))
    ids = ModelIndices.from_model(model, sim_cfg)
    joint_map = JointMap.from_model(model, sim_cfg, hw)
    safety = SafetyMonitor(model, ids, sim_cfg, safety_cfg)
    kinematics = PositionIKSolver(model, ids, sim_cfg, IKConfig())
    return LeRobotSo101Backend(bus, joint_map, safety, kinematics, model, sim_cfg, hw, clock, home_deg)

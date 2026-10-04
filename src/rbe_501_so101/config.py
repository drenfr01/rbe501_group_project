"""Central configuration: every tunable constant for simulation, tasks, and training."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


@dataclass(frozen=True)
class SimConfig:
    """MuJoCo model location, control-loop timing, home pose, and actuator limits."""

    model_path: Path = MODELS_DIR / "so101" / "so101.xml"
    n_substeps: int = 33
    initial_deg: tuple[float, ...] = (0.0, -20.0, 40.0, 15.0, 0.0, 40.0)
    stall_torque_nm: float = 2.9
    arm_joints: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    gripper_joint: str = "gripper"
    actuator_suffix: str = "_motor"
    tcp_site: str = "gripper_site"
    target_mocap_body: str = "reach_target"
    target_site: str = "reach_target_site"
    payload_body: str = "gripper_link"
    radial_axis_joint: str = "shoulder_pan"
    robot_root_body: str = "robot_mount"
    table_body: str = "table_frame"

    @property
    def joints(self) -> tuple[str, ...]:
        """All actuated joints: arm joints followed by the gripper."""
        return (*self.arm_joints, self.gripper_joint)


@dataclass(frozen=True)
class RenderConfig:
    """Offscreen image size and default viewer camera placement."""

    width: int = 640
    height: int = 480
    render_fps: int = 30
    lookat: tuple[float, float, float] = (0.02, 0.0, 0.5)
    distance: float = 1.05
    azimuth: float = 135.0
    elevation: float = -22.0


@dataclass(frozen=True)
class ReachTaskConfig:
    """Action scaling, episode length, and fixed gripper angle for the reach task."""

    delta_q_max_rad: float = 0.07
    max_episode_steps: int = 150
    gripper_hold_deg: float = 40.0


@dataclass(frozen=True)
class TargetSamplerConfig:
    """Clearance, radial band (from the shoulder-pan axis), and joint-limit margin."""

    z_min_m: float = 0.50
    radial_min_m: float = 0.12
    radial_max_m: float = 0.32
    joint_limit_margin_rad: float = 0.15
    max_attempts: int = 10_000


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the reach reward: -d + w * exp(-(d/sigma)^2) - c * ||a||^2."""

    bonus_weight: float = 1.0
    bonus_sigma_m: float = 0.005
    action_penalty: float = 0.01


@dataclass(frozen=True)
class DynamicsRandomizationConfig:
    """Per-episode actuator gain, damping, and payload randomization ranges."""

    enabled: bool = False
    kp_scale_range: tuple[float, float] = (0.7, 1.3)
    damping_scale_range: tuple[float, float] = (0.7, 1.3)
    payload_mass_range_kg: tuple[float, float] = (0.0, 0.05)


@dataclass(frozen=True)
class IKConfig:
    """Damped least-squares position IK solver settings.

    `seed_postures_deg` are (shoulder_lift, elbow_flex, wrist_flex) restart
    postures; each is combined with pan angles that face the target directly
    or from behind, so folded and over-the-top configurations are found.
    """

    tolerance_m: float = 0.001
    max_iterations: int = 500
    damping: float = 0.05
    step_gain: float = 0.5
    max_step_rad: float = 0.0872665
    seed_postures_deg: tuple[tuple[float, float, float], ...] = (
        (-20.0, 40.0, 15.0),
        (-75.0, -75.0, -75.0),
        (80.0, 80.0, 30.0),
    )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Headless throughput benchmark and viewer smoke-test settings."""

    throughput_steps: int = 10_000
    throughput_target_steps_per_s: float = 1_500.0
    smoke_episodes: int = 3
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """RL training hyperparameters, evaluation protocol, and success thresholds."""

    algo: str = "sac"
    timesteps: int = 300_000
    seeds: tuple[int, ...] = (0, 1, 2)
    learning_rate: float = 3e-4
    buffer_size: int = 300_000
    batch_size: int = 256
    gamma: float = 0.98
    model_dir: Path = field(default=CHECKPOINTS_DIR / "reach")
    eval_episodes: int = 20
    eval_seed_offset: int = 10_000
    holding_window_steps: int = 30
    success_mean_m: float = 0.005
    success_p95_m: float = 0.010
    holding_joint_speed_max_rad_s: float = 0.05
    model_filename: str = "model.zip"
    vecnormalize_filename: str = "vecnormalize.pkl"

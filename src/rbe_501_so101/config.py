"""Central configuration: every tunable constant for simulation, tasks, and training."""

from dataclasses import asdict, dataclass, field
from math import radians
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


@dataclass(frozen=True)
class SimConfig:
    """MuJoCo model location, control-loop timing, home pose, and actuator limits."""

    model_path: Path = MODELS_DIR / "so101" / "so101.xml"
    timestep_s: float = 0.001
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
    table_top_geom: str = "table_top"
    parked_bodies: tuple[str, ...] = ("cube",)

    @property
    def control_dt(self) -> float:
        """Duration of one policy step in seconds."""
        return self.n_substeps * self.timestep_s

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


ActionMode = Literal["delta", "residual_ik"]


@dataclass(frozen=True)
class ReachTaskConfig:
    """Action interface, episode length, observation extras, and gripper angle.

    `action_mode="delta"` adds `sign(a)|a|^action_exponent * delta_q_max_rad` to
    the joint command each step. `"residual_ik"` solves IK at reset and drives
    the command (at most `delta_q_max_rad` per step) toward the IK pose plus
    `residual_max_rad * sign(a)|a|^action_exponent`; the IK pose is observed.
    `fine_error_obs` appends the target error clipped to `fine_error_clip_m`
    and scaled to [-1, 1], so millimeter errors survive normalization.
    """

    delta_q_max_rad: float = 0.07
    max_episode_steps: int = 150
    gripper_hold_deg: float = 40.0
    action_mode: ActionMode = "delta"
    action_exponent: float = 1.0
    residual_max_rad: float = 0.1
    fine_error_obs: bool = False
    fine_error_clip_m: float = 0.02


@dataclass(frozen=True)
class TargetSamplerConfig:
    """Clearance, radial band (from the shoulder-pan axis), and joint-limit margin."""

    clearance_m: float = 0.025
    radial_min_m: float = 0.12
    radial_max_m: float = 0.32
    joint_limit_margin_rad: float = 0.15
    max_attempts: int = 10_000


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the reach reward.

    Distance is stored in meters and scaled by `distance_scale` (cm by default)
    so approaching the target has a usable gradient. The 5 mm bonus still only
    pays once sag is being corrected. `smoothness_penalty` damps action chatter.
    `precision_weight * ln(1 + d / precision_eps_m)` is steep at millimeter
    range, and `velocity_penalty * ||q_dot||^2` (faded out beyond about
    `velocity_radius_m`) rewards holding still at the target. Both default off.
    """

    bonus_weight: float = 1.0
    bonus_sigma_m: float = 0.005
    action_penalty: float = 0.01
    distance_scale: float = 100.0
    smoothness_penalty: float = 0.01
    precision_weight: float = 0.0
    precision_eps_m: float = 0.001
    velocity_penalty: float = 0.0
    velocity_radius_m: float = 0.02


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
    max_step_rad: float = radians(5.0)
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

    algo: Literal["sac", "ppo"] = "sac"
    timesteps: int = 300_000
    seeds: tuple[int, ...] = (0, 1, 2)
    learning_rate: float = 3e-4
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
    checkpoint_freq_steps: int = 25_000
    checkpoint_dirname: str = "checkpoint"
    checkpoint_log_filename: str = "checkpoint_log.json"
    replay_buffer_filename: str = "replay_buffer.pkl"
    run_config_filename: str = "run_config.json"
    monitor_stem: str = "train"


@dataclass(frozen=True)
class SacTuning:
    """Vectorization and SAC settings that vary between experiment presets.

    `gradient_steps` updates run after every vectorized step of `n_envs`
    transitions. `target_entropy` is SAC's entropy target (`"auto"` = -action
    dim); a lower value means less exploration noise once training settles.
    `norm_reward` rescales rewards with running return statistics.
    """

    n_envs: int = 1
    gradient_steps: int = 1
    buffer_size: int = 300_000
    target_entropy: float | Literal["auto"] = "auto"
    use_sde: bool = False
    norm_reward: bool = False


@dataclass(frozen=True)
class ExperimentPreset:
    """Everything that defines a training variant; saved next to each checkpoint."""

    task: ReachTaskConfig = field(default_factory=ReachTaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sac: SacTuning = field(default_factory=SacTuning)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable nested dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Inverse of `to_dict`."""
        return cls(
            task=ReachTaskConfig(**data["task"]),
            reward=RewardConfig(**data["reward"]),
            sac=SacTuning(**data["sac"]),
        )


PARALLEL_SAC = SacTuning(n_envs=4, gradient_steps=2, buffer_size=1_000_000)
LOW_ENTROPY_SAC = SacTuning(**{**asdict(PARALLEL_SAC), "target_entropy": -10.0})
SHAPED_REWARD = RewardConfig(precision_weight=1.0, velocity_penalty=2.0)
CUBIC_ACTION_EXPONENT = 3.0

PRESETS: dict[str, ExperimentPreset] = {
    "legacy": ExperimentPreset(),
    "control_long": ExperimentPreset(sac=PARALLEL_SAC),
    "shaped": ExperimentPreset(
        task=ReachTaskConfig(action_exponent=CUBIC_ACTION_EXPONENT, fine_error_obs=True),
        reward=SHAPED_REWARD,
        sac=LOW_ENTROPY_SAC,
    ),
    "residual_ik": ExperimentPreset(
        task=ReachTaskConfig(action_mode="residual_ik", fine_error_obs=True),
        reward=SHAPED_REWARD,
        sac=LOW_ENTROPY_SAC,
    ),
}


MassMode = Literal["fixed", "random"]


@dataclass(frozen=True)
class ObjectConfig:
    """Cube geometry for the pick-and-lift task."""

    body: str = "cube"
    geom: str = "cube_geom"
    side_m: float = 0.025


@dataclass(frozen=True)
class ObjectMassConfig:
    """Fixed mass or a uniform per-episode mass range, in kilograms."""

    mode: MassMode = "fixed"
    fixed_kg: float = 0.01
    range_kg: tuple[float, float] = (0.005, 0.02)

    def __post_init__(self) -> None:
        low, high = self.range_kg
        if self.fixed_kg <= 0.0 or low <= 0.0 or low > high:
            raise ValueError(f"masses must be positive with low <= high, got {self}")
        if self.mode not in ("fixed", "random"):
            raise ValueError(f"unknown mass mode {self.mode!r}")

    def sample(self, rng: np.random.Generator) -> float:
        """Choose one object mass for a new episode."""
        if self.mode == "fixed":
            return self.fixed_kg
        return float(rng.uniform(*self.range_kg))


@dataclass(frozen=True)
class GripperConfig:
    """Finger contact geometry and gripper angles."""

    fixed_jaw_mesh: str = "wrist_roll_follower_so101_v1"
    moving_jaw_mesh: str = "moving_jaw_so101_v1"
    fixed_pad_cutoff_m: float = -0.058
    moving_pad_cutoff_m: float = -0.048
    pad_friction: tuple[float, float, float] = (1.5, 0.01, 0.001)
    pad_solref: tuple[float, float] = (0.005, 1.0)
    pad_solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
    pinch_clearance_m: float = 0.004
    open_deg: float = 45.0
    close_deg: float = -10.0

    def pinch_point_b(self, side_m: float) -> tuple[float, float, float]:
        """Cube center offset in gripper coordinates."""
        return (-(side_m / 2 + self.pinch_clearance_m), 0.0, 0.0)


@dataclass(frozen=True)
class ContactSolverConfig:
    """Contact solver settings for the grasp model."""

    elliptic_cone: bool = True
    impratio: float = 10.0


@dataclass(frozen=True)
class ObjectPlacementConfig:
    """Cube spawn radius and azimuth around the pan axis."""

    radial_min_m: float = 0.14
    radial_max_m: float = 0.22
    azimuth_max_deg: float = 40.0


@dataclass(frozen=True)
class PickLiftTaskConfig:
    """Action scaling, episode length, and the lift height that counts as success."""

    delta_q_max_rad: float = 0.07
    delta_gripper_max_rad: float = 0.1
    max_episode_steps: int = 200
    lift_success_m: float = 0.04


@dataclass(frozen=True)
class PickLiftRewardConfig:
    """Weights for reach, grasp, lift, success, and action cost."""

    reach_scale_per_m: float = 10.0
    grasp_bonus: float = 0.25
    lift_weight: float = 1.0
    success_bonus: float = 1.0
    action_penalty: float = 0.01


@dataclass(frozen=True)
class PinchIKConfig:
    """Pinch-point IK tolerances, step size, and restart posture."""

    tolerance_m: float = 0.001
    tilt_tolerance_rad: float = radians(3.0)
    max_iterations: int = 600
    damping: float = 0.03
    step_gain: float = 0.5
    max_step_rad: float = 0.08
    seed_posture_deg: tuple[float, float, float] = (0.0, 0.0, 90.0)


@dataclass(frozen=True)
class ScriptedPickLiftConfig:
    """Grasp waypoints, phase lengths, and joint rates."""

    pregrasp_height_m: float = 0.05
    grasp_height_m: float = 0.0
    lift_height_m: float = 0.07
    approach_steps: int = 40
    descend_steps: int = 30
    close_steps: int = 30
    arm_rate_rad: float = 0.07
    close_rate_rad: float = 0.05
    lift_rate_rad: float = 0.02


@dataclass(frozen=True)
class ResidualConfig:
    """Residual control: executed action = scripted action + scale * policy action."""

    scale: float = 0.5


@dataclass(frozen=True)
class PickLiftEvalConfig:
    """Mass sweep used to compare fixed-mass and random-mass policies."""

    masses_kg: tuple[float, ...] = (0.005, 0.01, 0.015, 0.02)
    episodes_per_mass: int = 10
    seed_offset: int = 20_000


@dataclass(frozen=True)
class PickLiftConfig:
    """Settings for the pick-and-lift environment."""

    sim: SimConfig = field(
        default_factory=lambda: SimConfig(
            parked_bodies=(), initial_deg=(*SimConfig().initial_deg[:-1], GripperConfig().open_deg)
        )
    )
    task: PickLiftTaskConfig = field(default_factory=PickLiftTaskConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    object: ObjectConfig = field(default_factory=ObjectConfig)
    contact: ContactSolverConfig = field(default_factory=ContactSolverConfig)
    placement: ObjectPlacementConfig = field(default_factory=ObjectPlacementConfig)
    reward: PickLiftRewardConfig = field(default_factory=PickLiftRewardConfig)
    mass: ObjectMassConfig = field(default_factory=ObjectMassConfig)

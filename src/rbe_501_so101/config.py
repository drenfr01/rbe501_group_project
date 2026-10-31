"""Central configuration: every tunable constant for simulation, tasks, and training."""

from dataclasses import asdict, dataclass, field, replace
from math import pi, radians
from pathlib import Path
from typing import Any, Literal, Self

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"
ENCODER_COUNTS_PER_REV = 4096


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
    calibration_path: Path | None = None

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
    """Clearance, radial band (from the shoulder-pan axis), and joint-limit margin.

    `joint_window_rad` (sim radians, per arm joint) intersects the model's
    joint ranges (minus the margin) when given.
    """

    clearance_m: float = 0.025
    radial_min_m: float = 0.12
    radial_max_m: float = 0.32
    joint_limit_margin_rad: float = 0.15
    max_attempts: int = 10_000
    joint_window_rad: tuple[tuple[float, float], ...] | None = None


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
class PerturbationConfig:
    """Sensing and timing effects applied around a simulated backend.

    With `randomize_delay`, each episode draws its delay uniformly from
    `0..delay_steps` instead of using `delay_steps` exactly.
    """

    delay_steps: int = 0
    quantization_rad: float = 0.0
    randomize_delay: bool = False

    @property
    def active(self) -> bool:
        """True if any perturbation would change what the policy sees."""
        return self.delay_steps > 0 or self.quantization_rad > 0.0


ENCODER_QUANTIZATION_RAD = 2 * pi / ENCODER_COUNTS_PER_REV


@dataclass(frozen=True)
class ServoEffectsConfig:
    """Real-servo effects between the controller's command and the MuJoCo actuator.

    `stiction` applies the calibration's friction band as play on the servo
    goal: the goal only moves once the command pulls it further than the band,
    which reproduces the real arm's direction-dependent stop (D-03). Each
    episode scales the band by a draw from `stiction_scale_range`.
    `stiction_transition_ratio` rounds the corner of the play (0 = sharp play,
    1 = Dahl): after a reversal the goal starts slipping before the command has
    crossed the whole band. 1.5 fits the 3° and 8° sag checks and the pose
    check together (calibrated-sim gate, 2026-10-31).
    `relative_clip_rad` mimics LeRobot's `max_relative_target`: the goal
    reaching the servo stays within it of the measured position (None = off).
    """

    stiction: bool = False
    stiction_scale_range: tuple[float, float] = (1.0, 1.0)
    stiction_transition_ratio: float = 1.5
    relative_clip_rad: float | None = None

    @property
    def active(self) -> bool:
        """True if any effect changes what reaches the actuator."""
        return self.stiction or self.relative_clip_rad is not None


@dataclass(frozen=True)
class EnvOptions:
    """Simulator options a preset (or an evaluation) builds the reach env with.

    `calibration_path` is stored as a string so presets stay JSON-serializable.
    `randomization.enabled` forces randomization on even without the CLI flag.
    `target_joint_window_rad` (sim radians, per arm joint) further restricts the
    joint ranges reach targets are sampled from, e.g. to the real arm's range.
    """

    calibration_path: str | None = None
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    randomization: DynamicsRandomizationConfig = field(default_factory=DynamicsRandomizationConfig)
    servo: ServoEffectsConfig = field(default_factory=ServoEffectsConfig)
    target_joint_window_rad: tuple[tuple[float, float], ...] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Inverse of `asdict`, restoring tuples that JSON turned into lists (newer fields default)."""
        window = data.get("target_joint_window_rad")
        return cls(
            calibration_path=data["calibration_path"],
            perturbation=PerturbationConfig(**data["perturbation"]),
            randomization=DynamicsRandomizationConfig(**_tuples(data["randomization"])),
            servo=ServoEffectsConfig(**_tuples(data.get("servo", {}))),
            target_joint_window_rad=None if window is None else tuple(tuple(w) for w in window),
        )


def _tuples(data: dict[str, Any]) -> dict[str, Any]:
    """Lists (from JSON) back to tuples."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}


@dataclass(frozen=True)
class ExperimentPreset:
    """Everything that defines a training variant; saved next to each checkpoint."""

    task: ReachTaskConfig = field(default_factory=ReachTaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sac: SacTuning = field(default_factory=SacTuning)
    env: EnvOptions = field(default_factory=EnvOptions)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable nested dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Inverse of `to_dict`; run configs saved before `env` existed get the defaults."""
        env = EnvOptions.from_dict(data["env"]) if "env" in data else EnvOptions()
        return cls(
            task=ReachTaskConfig(**data["task"]),
            reward=RewardConfig(**data["reward"]),
            sac=SacTuning(**data["sac"]),
            env=env,
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

CALIBRATION_DIR = DATA_DIR / "calibration"
DEFAULT_CALIBRATION_FILE = "data/calibration/actuator_calibration.json"
CALIBRATED_RANDOMIZATION = DynamicsRandomizationConfig(enabled=True)
CALIBRATED_PERTURBATION = PerturbationConfig(
    delay_steps=1, quantization_rad=ENCODER_QUANTIZATION_RAD, randomize_delay=True
)
PRESETS["residual_ik_calibrated"] = ExperimentPreset(
    task=PRESETS["residual_ik"].task,
    reward=PRESETS["residual_ik"].reward,
    sac=PRESETS["residual_ik"].sac,
    env=EnvOptions(
        calibration_path=DEFAULT_CALIBRATION_FILE,
        perturbation=CALIBRATED_PERTURBATION,
        randomization=CALIBRATED_RANDOMIZATION,
    ),
)


def resolve_project_path(path: str | Path) -> Path:
    """Absolute path; relative paths are taken from the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class ProvenanceConfig:
    """Run-manifest file names and the packages whose versions are recorded."""

    manifest_filename: str = "manifest.json"
    patch_filename: str = "uncommitted.patch"
    packages: tuple[str, ...] = ("rbe_501_so101", "mujoco", "lerobot", "torch")


@dataclass(frozen=True)
class JointMappingEntry:
    """How one LeRobot joint reading maps onto the matching sim joint."""

    sign: float = 1.0
    offset_rad: float = 0.0


@dataclass(frozen=True)
class HardwareConfig:
    """Physical SO-101 connection, unit mapping, and motion limits.

    `position_p_gain` is the STS3215 `P_Coefficient` LeRobot writes on connect
    (LeRobot default 16, Feetech factory default 32). 24 was chosen from the
    2026-10-31 gain sweep: steady-state error scales as 1/P, and 32 was jerky
    in motion. Every actuator calibration is specific to the gain it was
    measured at.

    `real_half_range_deg` is each arm joint's calibrated half range in LeRobot
    degrees, from LeRobot's `rbe501_follower` calibration file (2026-10-31); the
    live value comes from the bus, this copy lets training use the window offline.

    `rest_deg` is in sim degrees (arm joints, then gripper): the pose ramped to
    before torque is cut. It is the folded, supported pose read from the arm
    on 2026-10-31 (LeRobot -2.42, -73.19, 96.57, 3.25, 2.24, gripper 0.27).
    """

    port: str = "/dev/so101_follower"
    robot_id: str = "rbe501_follower"
    max_relative_target_deg: float = 10.0
    position_p_gain: int = 24
    position_p_gain_range: tuple[int, int] = (1, 254)
    speed_scale: float = 0.5
    home_ramp_s: float = 3.0
    settle_s: float = 1.0
    limit_margin_rad: float = 0.15
    joint_map: tuple[JointMappingEntry, ...] = (
        JointMappingEntry(1.0, radians(1.5)),
        JointMappingEntry(1.0, radians(-26.8)),
        JointMappingEntry(1.0, radians(-6.7)),
        JointMappingEntry(1.0, radians(-0.5)),
        JointMappingEntry(1.0, 0.0),
    )
    gripper_percent_range: tuple[float, float] = (0.0, 100.0)
    mapping_tolerance_deg: float = 0.5
    rest_deg: tuple[float, ...] = (-0.92, -99.99, 89.87, 2.75, 2.24, -9.7)
    rest_ramp_s: float = 3.0
    lerobot_degrees_per_count: float = 360.0 / 4095
    real_half_range_deg: tuple[float, ...] = (118.7, 87.8, 97.0, 104.9, 180.0)

    @property
    def max_command_step_rad(self) -> float:
        """Largest per-step command change deployed on the arm: the default sim rate times `speed_scale`."""
        return self.speed_scale * ReachTaskConfig().delta_q_max_rad

    def sim_joint_window_rad(self) -> tuple[tuple[float, float], ...]:
        """Real calibrated arm range in sim radians, shrunk by `limit_margin_rad` (what the real backend allows)."""
        window = []
        for half, entry in zip(self.real_half_range_deg, self.joint_map, strict=True):
            ends = sorted(entry.sign * radians(s * half) + entry.offset_rad for s in (-1.0, 1.0))
            window.append((ends[0] + self.limit_margin_rad, ends[1] - self.limit_margin_rad))
        return tuple(window)


# Plan steps 8-12: train on the arm as measured on 2026-10-31. Stiffness from the
# 8° sag check, randomized 0.7-1.4x around it (pan and roll kp are assumed);
# stiction randomized 0.5-1.5x around the fitted band; LeRobot's 10° relative
# clip; reach targets inside the real joint range; the real arm's command rate,
# so deployment needs no slowdown; a 0.2 rad residual, enough for sag plus a
# friction band at the soft end of the stiffness range.
SIM2REAL_RESIDUAL_MAX_RAD = 0.2
SIM2REAL_KP_SCALE_RANGE = (0.7, 1.4)
SIM2REAL_STICTION_SCALE_RANGE = (0.5, 1.5)
PRESETS["residual_ik_sim2real"] = ExperimentPreset(
    task=replace(
        PRESETS["residual_ik"].task,
        delta_q_max_rad=HardwareConfig().max_command_step_rad,
        residual_max_rad=SIM2REAL_RESIDUAL_MAX_RAD,
    ),
    reward=PRESETS["residual_ik"].reward,
    sac=PRESETS["residual_ik"].sac,
    env=EnvOptions(
        calibration_path=DEFAULT_CALIBRATION_FILE,
        perturbation=CALIBRATED_PERTURBATION,
        randomization=replace(CALIBRATED_RANDOMIZATION, kp_scale_range=SIM2REAL_KP_SCALE_RANGE),
        servo=ServoEffectsConfig(
            stiction=True,
            stiction_scale_range=SIM2REAL_STICTION_SCALE_RANGE,
            relative_clip_rad=radians(HardwareConfig().max_relative_target_deg),
        ),
        target_joint_window_rad=HardwareConfig().sim_joint_window_rad(),
    ),
)


@dataclass(frozen=True)
class SafetyConfig:
    """Abort thresholds checked on every hardware control step."""

    max_temperature_c: float = 55.0
    max_load_fraction: float = 0.8
    max_consecutive_overruns: int = 3
    table_clearance_m: float = 0.02
    status_check_every_steps: int = 10
    clearance_root_body: str = "upper_arm_link"


@dataclass(frozen=True)
class PathCheckConfig:
    """Pre-flight table check of a joint-space path before torque is enabled.

    Points are checked every `step_deg`, both as commanded and with
    `sag_allowance_deg` added (positive sim lift/elbow/wrist move the arm down).
    """

    step_deg: float = 1.0
    sag_allowance_deg: tuple[float, ...] = (0.0, 6.0, 6.0, 6.0, 0.0)


@dataclass(frozen=True)
class SagCheckConfig:
    """Phase 0: how each recorded pose is approached and sampled.

    Ramps move every joint at most `ramp_speed_deg_s`, sending a goal every
    `ramp_period_s`. "From above" approaches each loaded joint from the side
    gravity pulls away from (lifting the links past it); joints holding less
    than `gravity_torque_floor_nm` use the positive LeRobot side. Every pose is
    entered from `waypoint_sim_deg` (sim degrees, arm joints), and the whole
    path is checked against the table before torque is enabled.
    `approach_offset_range_deg` bounds a command-line override of the offset
    (larger offsets sweep the whole friction band but move closer to limits).
    """

    poses_path: Path = DATA_DIR / "sag_check" / "poses" / "poses.json"
    sessions_dir: Path = DATA_DIR / "sag_check"
    session_filename: str = "session.json"
    approach_offset_deg: float = 3.0
    approach_offset_range_deg: tuple[float, float] = (0.5, 15.0)
    gravity_torque_floor_nm: float = 0.01
    waypoint_sim_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    path_check: PathCheckConfig = PathCheckConfig()
    settle_s: float = 2.0
    n_reads: int = 20
    read_interval_s: float = 0.02
    ramp_speed_deg_s: float = 30.0
    ramp_period_s: float = 1.0 / 30.0
    clamp_tolerance_deg: float = 1e-6
    final_goal_timeout_s: float = 3.0


@dataclass(frozen=True)
class SagFitConfig:
    """Stiffness fit: confidence level and when a joint counts as identifiable.

    A joint is not identifiable when its gravity torque spans less than
    `min_torque_span_nm` across the poses.

    Stiction (the friction band) is only fitted from sessions whose approach
    offsets are all at least `min_friction_approach_deg`: smaller approaches
    land inside the band and under-measure it (3° gave a third of the 8°
    band on 2026-10-31). The band's load slope is fitted only when the
    joint's gravity torque spans `min_slope_torque_span_nm`; otherwise the
    band is the mean. With `assume_unidentified_kp`, joints without gravity
    load (pan, wrist roll) get the mean fitted stiffness, flagged as assumed:
    every arm joint is the same servo at the same gain.
    """

    confidence: float = 0.95
    min_torque_span_nm: float = 0.05
    min_samples: int = 3
    early_warning_fraction: float = 0.5
    min_hysteresis_deg: float = 360.0 / ENCODER_COUNTS_PER_REV
    min_friction_approach_deg: float = 6.0
    min_slope_torque_span_nm: float = 0.2
    assume_unidentified_kp: bool = True
    experiment_suffix: str = "sag_analysis"


@dataclass(frozen=True)
class CheckPose:
    """A pose in sim degrees (arm joints) and what the operator measures on the physical arm.

    At sim zero the upper-arm mesh is vertical and the forearm and gripper are
    level, so angles in `check` are measured on the links themselves.
    """

    name: str
    sim_deg: tuple[float, ...]
    check: str


DEFAULT_CHECK_POSES: tuple[CheckPose, ...] = (
    CheckPose("L_pose", (0.0, 0.0, 0.0, 0.0, 0.0), "upper arm vertical (90°), forearm level (0°), gripper level (0°)"),
    CheckPose("forearm_up_30", (0.0, 0.0, -30.0, 0.0, 0.0), "upper arm vertical, forearm and gripper 30° above level"),
    CheckPose("lean_forward_30", (0.0, 30.0, -30.0, 0.0, 0.0), "upper arm 30° forward of vertical, forearm and gripper level"),
    CheckPose("wrist_down_45", (0.0, 0.0, 0.0, 45.0, 0.0), "upper arm vertical, forearm level, gripper 45° below level"),
    CheckPose("pan_45", (45.0, 0.0, 0.0, 0.0, 0.0), "L pose with the base turned 45° from straight ahead"),
)


@dataclass(frozen=True)
class PoseCheckConfig:
    """Pose check: command known sim poses on the real arm and compare with level measurements.

    Before torque is enabled the whole path (start, every pose, back to start)
    is checked against the table (`path_check`) before torque is enabled.
    """

    poses: tuple[CheckPose, ...] = DEFAULT_CHECK_POSES
    results_dir: Path = DATA_DIR / "pose_check"
    results_filename: str = "results.json"
    ramp_speed_deg_s: float = 15.0
    ramp_period_s: float = 1.0 / 30.0
    settle_s: float = 2.0
    n_reads: int = 20
    read_interval_s: float = 0.02
    clamp_tolerance_deg: float = 1e-6
    final_goal_timeout_s: float = 3.0
    path_check: PathCheckConfig = PathCheckConfig()


@dataclass(frozen=True)
class CalibratedSimGateConfig:
    """Plan step 10: the calibrated sim must reproduce the real measurements before training.

    For every joint and arrival direction, the mean absolute difference
    between real and replayed error must stay within `tolerance_deg` (about
    2 mm at the gripper for the shoulder at reach distance).
    """

    tolerance_deg: float = 1.0
    experiment_suffix: str = "calibrated_sim_gate"


@dataclass(frozen=True)
class MirrorConfig:
    """Phase 1 mirror tool loop rate and where reference readings are saved."""

    loop_hz: float = 30.0
    references_dir: Path = DATA_DIR / "joint_mapping"
    references_filename: str = "references.json"


@dataclass(frozen=True)
class IntegralControlConfig:
    """Model-free baseline: fraction of the remaining joint error added to the command each step.

    Integration starts once the command is within `reached_tol_rad` of the IK pose.
    """

    gain: float = 0.2
    reached_tol_rad: float = 1e-4


@dataclass(frozen=True)
class RobustnessConfig:
    """Phase 2 sweep grid and the transfer-or-retrain thresholds."""

    kp_scales: tuple[float, ...] = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    delay_steps: tuple[int, ...] = (0, 1)
    quantization_rad: float = ENCODER_QUANTIZATION_RAD
    episodes_per_point: int = 20
    saturation_threshold: float = 0.95
    transfer_max_mean_m: float = 0.002
    max_saturation_fraction: float = 0.05
    experiment_suffix: str = "reach_robustness"


@dataclass(frozen=True)
class AcceptanceConfig:
    """Plan step 15: what a retrained policy must do in the calibrated sim before it goes on the arm.

    Judged in the deployment condition (every random draw at the measured arm,
    the longest training delay, encoder rounding) and in stress conditions at
    the edges of the training randomization (each kp extreme, the largest
    stiction band). The policy must hold within `deployment_max_mean_mm`, beat
    IK hold by more than its seed spread, saturate on less than
    `max_saturation_fraction` of holding steps, stay within
    `worst_case_max_mean_mm` under stress, and be no worse than integral
    control (plus `integral_margin_mm`) in any condition. Settling time and the
    two-sided approach test are left to the hardware repeatability runs.
    """

    deployment_max_mean_mm: float = 2.0
    worst_case_max_mean_mm: float = 5.0
    max_saturation_fraction: float = 0.2
    integral_margin_mm: float = 0.0
    kp_scales: tuple[float, ...] = SIM2REAL_KP_SCALE_RANGE
    stiction_scales: tuple[float, ...] = (SIM2REAL_STICTION_SCALE_RANGE[1],)
    episodes_per_condition: int = 20
    experiment_suffix: str = "acceptance"


@dataclass(frozen=True)
class RealTrialConfig:
    """Phase 3 output naming and the fixed repeatability protocol."""

    experiment_suffix: str = "real_reach"
    repeat_targets: int = 3
    repeats_per_target: int = 10
    fixed_targets_path: Path = DATA_DIR / "targets" / "reach_fixed.json"
    repeat_targets_path: Path = DATA_DIR / "targets" / "reach_repeat.json"
    n_fixed_targets: int = 10

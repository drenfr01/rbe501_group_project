"""Explicit calibrated residual-lift preset, independent of legacy training defaults."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rbe_501_so101.config import (
    DEFAULT_CALIBRATION_FILE, PRESETS, PROJECT_ROOT, ObjectMassConfig, PickLiftConfig,
    ResidualConfig, ScriptedPickLiftConfig, TrainConfig, resolve_project_path,
)
from rbe_501_so101.control.aligned_pick_lift import AlignedPickLift, MotionOptions
from rbe_501_so101.control.guarded_pick_lift import GuardedLift
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.envs.model_indices import descendant_body_ids
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from rbe_501_so101.rl.pick_lift import NormalizedPolicy
from rbe_501_so101.rl.residual import ResidualPickLift


def outer_wrist_motion() -> MotionOptions:
    """The previously validated controller, with an explicit outer-reach threshold."""
    return MotionOptions("outer_wrist", gravity_feedforward=True, radial_jaw_alignment=True,
                         grasp_height_m=-.004, opposite_wrist_radius_m=.21)


@dataclass(frozen=True)
class CalibratedLiftPreset:
    """Serialize controller, arm effects, lift guard and mass distribution together."""

    name: str = "calibrated_outer_wrist_v1"
    motion: MotionOptions = field(default_factory=outer_wrist_motion)
    mass: ObjectMassConfig = field(default_factory=lambda: ObjectMassConfig(mode="random", range_kg=(.005, .02)))
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    arm_effects: bool = True
    stable_steps: int = 5
    calibration_file: str = DEFAULT_CALIBRATION_FILE

    def environment(self) -> PickLiftConfig:
        """Build the unchanged task with measured stiffness and the recorded mass rule."""
        base = PickLiftConfig()
        sim = replace(base.sim, calibration_path=resolve_project_path(self.calibration_file))
        return replace(base, sim=sim, mass=self.mass)


def canonical(value: object) -> object:
    """JSON-compatible snapshot, with portable project-relative paths."""
    def path_string(path: Path) -> str:
        """Avoid encoding this PC's project directory into the configuration."""
        return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)

    return json.loads(json.dumps(value, default=path_string))


def snapshot(preset: CalibratedLiftPreset) -> object:
    """Record resolved defaults too, so a later code change cannot silently alter a run."""
    options = PRESETS["residual_ik_sim2real_smooth"].env
    return canonical({"preset": asdict(preset), "environment": asdict(preset.environment()),
                      "scripted": asdict(ScriptedPickLiftConfig()),
                      "servo": asdict(replace(options.servo, stiction_scale_range=(1., 1.))),
                      "sensing": asdict(replace(options.perturbation, randomize_delay=False)),
                      "effects_rng_seed": 0, "wrapper_order": ["residual", "contact_audit", "guard", "raw"]})


def input_hashes(preset: CalibratedLiftPreset) -> dict[str, str]:
    """Fingerprint the model and calibration required to reconstruct this preset."""
    paths = (preset.environment().sim.model_path, resolve_project_path(preset.calibration_file))
    return {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


@dataclass(frozen=True)
class LiftPresetRepository:
    """Own the immutable run configuration beside a fresh pilot checkpoint directory."""

    root: Path

    @property
    def path(self) -> Path:
        """Serialized configuration used for both training and evaluation."""
        return self.root / "run_config.json"

    def create(self, preset: CalibratedLiftPreset, training: TrainConfig, seed: int, steps: int) -> None:
        """Refuse any existing directory rather than silently resuming another experiment."""
        self.root.mkdir(parents=True, exist_ok=False)
        record = {"schema": 1, "preset": asdict(preset), "snapshot": snapshot(preset),
                  "training": asdict(training), "seed": seed, "steps": steps,
                  "input_sha256": input_hashes(preset)}
        self.path.write_text(json.dumps(canonical(record), indent=2), encoding="utf-8")

    def load(self) -> CalibratedLiftPreset:
        """Reject configuration/input drift before reconstructing a policy environment."""
        record = json.loads(self.path.read_text(encoding="utf-8"))
        raw = record["preset"]
        mass = {**raw["mass"], "range_kg": tuple(raw["mass"]["range_kg"])}
        preset = CalibratedLiftPreset(**{**raw, "motion": MotionOptions(**raw["motion"]),
                                         "mass": ObjectMassConfig(**mass), "residual": ResidualConfig(**raw["residual"])})
        if record["schema"] != 1 or record["snapshot"] != snapshot(preset):
            raise ValueError("Preset snapshot differs from this runtime configuration")
        if record["input_sha256"] != input_hashes(preset):
            raise ValueError("Model or calibration checksum changed")
        return preset


class ContactAudit(gym.Wrapper):
    """Count robot/table contacts at control boundaries in each training episode."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        raw = self.unwrapped
        self.bodies = descendant_body_ids(raw.backend.model, raw.config.sim.robot_root_body)
        self.table = raw.backend.model.body(raw.config.sim.table_body).id
        self.contacts = 0

    def reset(self, **kwargs):
        """Start a fresh per-episode contact counter."""
        self.contacts = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Expose cumulative contacts and warnings alongside existing task information."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = self.unwrapped
        for contact in raw.backend.data.contact:
            pair = {int(raw.backend.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
            if self.table in pair and pair & self.bodies:
                self.contacts += 1
                break
        info.update(robot_table_contact_steps=self.contacts, mujoco_warning=raw.backend.has_warnings())
        return obs, reward, terminated, truncated, info


def make_env(preset: CalibratedLiftPreset, render_mode=None) -> gym.Env:
    """Construct the validated stack; the guard filters the final residual action."""
    raw = SO101PickLiftEnv(cfg=preset.environment(), render_mode=render_mode)
    controller = AlignedPickLift.for_env(raw, preset.motion)
    if preset.arm_effects:
        raw.backend = GraspServoBackend(raw.backend, raw.config)
    guarded = ContactAudit(GuardedLift(raw, preset.stable_steps))
    return ResidualPickLift(guarded, controller, preset.residual)


def factory(preset: CalibratedLiftPreset, monitor: Path | None = None) -> Callable[[], gym.Env]:
    """Reuse the same explicit environment definition for training and normalization."""
    def create() -> gym.Env:
        """Monitor episode rewards, success, lift, guard state and table contacts."""
        keys = ("is_success", "mass_kg", "lift_m", "robot_table_contact_steps",
                "mujoco_warning", "lift_released", "grasp_abort")
        return Monitor(make_env(preset), filename=str(monitor) if monitor else None, info_keywords=keys)

    return create


def load_policy(repo: LiftPresetRepository, seed: int = 0) -> NormalizedPolicy:
    """Load the pilot with its matching frozen statistics and verified preset."""
    preset = repo.load()
    venv = DummyVecEnv([factory(preset)])
    seed_dir = repo.root / f"seed_{seed}"
    normalizer = VecNormalize.load(str(seed_dir / "vecnormalize.pkl"), venv)
    normalizer.training, normalizer.norm_reward = False, False
    model = SAC.load(seed_dir / "model.zip", device="cpu")
    return NormalizedPolicy(model, normalizer)

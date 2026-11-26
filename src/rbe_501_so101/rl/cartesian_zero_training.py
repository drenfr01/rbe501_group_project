"""Versioned RL preset for the validated Cartesian acquisition and 0-degree jaw."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rbe_501_so101.config import (
    ObjectMassConfig, PinchIKConfig, ResidualConfig, ScriptedPickLiftConfig, TrainConfig,
)
from rbe_501_so101.control.aligned_pick_lift import MotionOptions
from rbe_501_so101.control.cartesian_descent import CartesianDescent, CartesianGuard, DescentConfig
from rbe_501_so101.control.gripper_closing import ClosingProfile
from rbe_501_so101.control.raised_approach import ApproachConfig
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from rbe_501_so101.rl.calibrated_pick_lift import (
    CalibratedLiftPreset, ContactAudit, canonical, input_hashes, outer_wrist_motion,
    snapshot as base_snapshot,
)
from rbe_501_so101.rl.lift_residual_limits import LimitedLiftResidual, ResidualLimits
from rbe_501_so101.rl.pick_lift import NormalizedPolicy


def calibrated_base() -> CalibratedLiftPreset:
    """Keep the reaching policy separate; configure only the calibrated pick/lift task."""
    return CalibratedLiftPreset(name="calibrated_cartesian_zero_v3",
                               motion=replace(outer_wrist_motion(), grasp_height_m=0.),
                               residual=ResidualConfig(scale=.05))


@dataclass(frozen=True)
class CartesianZeroPreset:
    """One serialized definition of acquisition, closing, arm effects and RL authority."""

    base: CalibratedLiftPreset = field(default_factory=calibrated_base)
    descent: DescentConfig = field(default_factory=lambda: DescentConfig(name="cartesian_table", check_cube=False))
    closing: ClosingProfile = field(default_factory=lambda: ClosingProfile("position_zero", target_deg=0.))
    limits: ResidualLimits = field(default_factory=lambda: ResidualLimits(clearance_m=.004))

    def __post_init__(self) -> None:
        """Reject inconsistent or unsupported combinations instead of silently changing control."""
        if self.closing.stop_on_contact or self.base.residual.scale != self.limits.scale:
            raise ValueError("Fixed closing endpoint and matching residual scales required")
        if self.base.motion.grasp_height_m != self.descent.grasp_height_m:
            raise ValueError("Motion and descent grasp heights must match")

    def environment(self):
        """Only replace the controller's closing setpoint; compiled physics stays unchanged."""
        cfg = self.base.environment()
        return replace(cfg, gripper=replace(cfg.gripper, close_deg=self.closing.target_deg))

    def resolved_descent(self) -> DescentConfig:
        """Use the same closing dwell calculation as the validated scripted builder."""
        travel = np.deg2rad(self.environment().gripper.open_deg - self.closing.target_deg)
        steps = int(np.ceil(travel / self.closing.rate_rad)) + self.base.stable_steps
        return replace(self.descent, close_dwell_steps=max(self.descent.close_dwell_steps, steps))


def snapshot(preset: CartesianZeroPreset) -> object:
    """Capture effective configuration, including inherited defaults and wrapper ordering."""
    record = dict(base_snapshot(preset.base))
    record.update(preset=asdict(preset), environment=asdict(preset.environment()),
                  scripted=asdict(replace(ScriptedPickLiftConfig(), grasp_height_m=preset.descent.grasp_height_m,
                                         close_rate_rad=preset.closing.rate_rad)),
                  acquisition=asdict(ApproachConfig("cartesian", xy_tolerance_m=preset.descent.xy_tolerance_m,
                                                   z_tolerance_m=preset.descent.final_z_tolerance_m)),
                  descent=asdict(preset.resolved_descent()), ik=asdict(PinchIKConfig()),
                  limits=asdict(preset.limits), shadow_reference="starts_after_actual_guard_release",
                  wrapper_order=["limited_cartesian_residual", "contact_audit", "cartesian_guard", "raw"])
    return canonical(record)


def decode(raw: dict) -> CartesianZeroPreset:
    """Restore nested dataclasses and tuple-valued mass settings from JSON."""
    base = raw["base"]
    mass = {**base["mass"], "range_kg": tuple(base["mass"]["range_kg"])}
    restored = CalibratedLiftPreset(**{**base, "motion": MotionOptions(**base["motion"]),
                                      "mass": ObjectMassConfig(**mass), "residual": ResidualConfig(**base["residual"])})
    return CartesianZeroPreset(restored, DescentConfig(**raw["descent"]),
                               ClosingProfile(**raw["closing"]), ResidualLimits(**raw["limits"]))


@dataclass(frozen=True)
class CartesianZeroRepository:
    """Own a fresh v3 run and refuse drift in physics or controller reconstruction."""

    root: Path

    @property
    def path(self) -> Path:
        """The run's full training/environment configuration."""
        return self.root / "run_config.json"

    def create(self, preset: CartesianZeroPreset, training: TrainConfig, seed: int, steps: int) -> None:
        """Never resume or overwrite an older checkpoint as part of this pilot."""
        self.root.mkdir(parents=True, exist_ok=False)
        record = {"schema": 3, "preset": asdict(preset), "snapshot": snapshot(preset),
                  "training": asdict(training), "seed": seed, "steps": steps,
                  "input_sha256": input_hashes(preset.base)}
        self.path.write_text(json.dumps(canonical(record), indent=2), encoding="utf-8")

    def load(self) -> CartesianZeroPreset:
        """Require the saved snapshot and model/calibration bytes to match this runtime."""
        record = json.loads(self.path.read_text(encoding="utf-8"))
        if record["schema"] != 3:
            raise ValueError("Expected v3 Cartesian-zero snapshot")
        preset = decode(record["preset"])
        if record["snapshot"] != snapshot(preset):
            raise ValueError("Cartesian-zero snapshot differs from runtime configuration")
        if record["input_sha256"] != input_hashes(preset.base):
            raise ValueError("Model or calibration checksum changed")
        return preset


class CartesianLiftResidual(LimitedLiftResidual):
    """Keep the shadow command reference stationary during guarded closure settling."""

    def _advance_reference(self) -> None:
        """Do not project residuals around a fictitious lift that has not executed yet."""
        gate = self.env.env.gate
        if not gate.released or gate.aborted:
            self.reference = None
            return
        super()._advance_reference()


def make_env(preset: CartesianZeroPreset, render_mode=None) -> CartesianLiftResidual:
    """Reconstruct the tested scripted sequence before adding lift-only RL corrections."""
    raw = SO101PickLiftEnv(cfg=preset.environment(), render_mode=render_mode)
    controller = CartesianDescent.for_env(raw, preset.resolved_descent())
    controller.options = preset.base.motion
    controller._cfg = replace(controller._cfg, close_rate_rad=preset.closing.rate_rad)
    if preset.base.arm_effects:
        raw.backend = GraspServoBackend(raw.backend, raw.config)
    guard = CartesianGuard(raw, controller)
    guard.stable_steps = preset.base.stable_steps
    return CartesianLiftResidual(ContactAudit(guard), controller, preset.limits)


def factory(preset: CartesianZeroPreset, monitor: Path | None = None) -> Callable[[], gym.Env]:
    """Monitor task outcome, table contacts and the residual's actual intervention counts."""
    def create() -> gym.Env:
        """Build a new monitored v3 environment for SB3."""
        keys = ("is_success", "mass_kg", "lift_m", "robot_table_contact_steps", "mujoco_warning",
                "lift_released", "grasp_abort", "active_steps", "applied_steps",
                "clearance_rejections", "holding_rejections", "budget_clips", "applied_norm_sum")
        return Monitor(make_env(preset), filename=str(monitor) if monitor else None, info_keywords=keys)

    return create


def load_policy(repo: CartesianZeroRepository, seed: int = 0) -> NormalizedPolicy:
    """Load only the v3 checkpoint with its own preset and frozen normalization state."""
    vector = DummyVecEnv([factory(repo.load())])
    seed_dir = repo.root / f"seed_{seed}"
    normalizer = VecNormalize.load(str(seed_dir / "vecnormalize.pkl"), vector)
    normalizer.training, normalizer.norm_reward = False, False
    return NormalizedPolicy(SAC.load(seed_dir / "model.zip", device="cpu"), normalizer)

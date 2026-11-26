"""Separate serialized v2 preset; legacy calibrated presets and policies stay intact."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rbe_501_so101.config import ResidualConfig, TrainConfig
from rbe_501_so101.rl.calibrated_pick_lift import (
    CalibratedLiftPreset, LiftPresetRepository, canonical, make_env as legacy_env,
)
from rbe_501_so101.rl.lift_residual_limits import LimitedLiftResidual, ResidualLimits
from rbe_501_so101.rl.pick_lift import NormalizedPolicy


@dataclass(frozen=True)
class LimitedLiftPreset:
    """Tie the validated base controller to a separately versioned residual envelope."""

    base: CalibratedLiftPreset = field(default_factory=lambda: CalibratedLiftPreset(
        name="calibrated_outer_wrist_lift_only_v2", residual=ResidualConfig(scale=.05)))
    limits: ResidualLimits = field(default_factory=ResidualLimits)


@dataclass(frozen=True)
class LimitedLiftRepository:
    """Store v2 constraints alongside the existing full environment snapshot."""

    root: Path

    @property
    def path(self) -> Path:
        """One configuration file for both base physics and new correction limits."""
        return self.root / "run_config.json"

    def create(self, preset: LimitedLiftPreset, training: TrainConfig, seed: int, steps: int) -> None:
        """Create a new directory; refuse existing checkpoints and inconsistent scales."""
        if preset.base.residual.scale != preset.limits.scale:
            raise ValueError("Residual scale must match the serialized envelope")
        LiftPresetRepository(self.root).create(preset.base, training, seed, steps)
        record = json.loads(self.path.read_text())
        record.update(limited_schema=1, limits=asdict(preset.limits),
                      limits_snapshot=asdict(preset.limits),
                      limited_wrapper_order=["limited_lift_residual", "contact_audit", "guard", "raw"])
        self.path.write_text(json.dumps(canonical(record), indent=2), encoding="utf-8")

    def load(self) -> LimitedLiftPreset:
        """Load the exact saved envelope rather than substituting current defaults."""
        record = json.loads(self.path.read_text())
        if record.get("limited_schema") != 1 or record["limits"] != record["limits_snapshot"]:
            raise ValueError("Missing or changed limited-residual snapshot")
        base = LiftPresetRepository(self.root).load()
        limits = ResidualLimits(**record["limits"])
        if base.residual.scale != limits.scale:
            raise ValueError("Residual scale does not match envelope")
        return LimitedLiftPreset(base, limits)


def make_env(preset: LimitedLiftPreset, render_mode=None) -> LimitedLiftResidual:
    """Reuse the validated physics/script/guard, replacing only residual authority."""
    legacy = legacy_env(preset.base, render_mode)
    return LimitedLiftResidual(legacy.env, legacy._controller, preset.limits)


def factory(preset: LimitedLiftPreset, monitor: Path | None = None) -> Callable[[], gym.Env]:
    """Monitor success, contacts and interventions without changing the task reward."""
    def create() -> gym.Env:
        """Create one fresh constrained training environment."""
        keys = ("is_success", "mass_kg", "lift_m", "robot_table_contact_steps", "mujoco_warning",
                "lift_released", "grasp_abort", "active_steps", "applied_steps",
                "clearance_rejections", "holding_rejections", "budget_clips", "applied_norm_sum")
        return Monitor(make_env(preset), filename=str(monitor) if monitor else None, info_keywords=keys)

    return create


def load_policy(repo: LimitedLiftRepository, seed: int = 0) -> NormalizedPolicy:
    """Reload the v2 policy with its own verified environment and frozen statistics."""
    preset = repo.load()
    vector = DummyVecEnv([factory(preset)])
    seed_dir = repo.root / f"seed_{seed}"
    normalizer = VecNormalize.load(str(seed_dir / "vecnormalize.pkl"), vector)
    normalizer.training, normalizer.norm_reward = False, False
    model = SAC.load(seed_dir / "model.zip", device="cpu")
    return NormalizedPolicy(model, normalizer)

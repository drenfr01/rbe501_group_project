"""Compare original versus measured-stiffness-only lift dynamics, without training.

Run from the project root:
    uv run python scripts/sim2real/analyze/pick_lift_calibration.py --episodes 20 --out experiments/pick_lift_stiffness_check

Uses the same placement seeds at 5, 10, 15, and 20 g for the scripted baseline
and both shared seed-0 policies. Contacts, task settings, and policy weights
stay fixed. This does NOT add the reaching stiction/delay/quantization wrappers.
The new output directory must not already exist; checkpoints are only read.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.config import (
    DEFAULT_CALIBRATION_FILE, PICK_LIFT_CHECKPOINTS_DIR, ObjectMassConfig,
    PickLiftConfig, PickLiftEvalConfig, TrainConfig, resolve_project_path,
)
from rbe_501_so101.control import ScriptedPickLift
from rbe_501_so101.envs.model_indices import descendant_body_ids
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from rbe_501_so101.rl.pick_lift import EpisodePolicy, load_policy, load_run
from rbe_501_so101.rl.residual import ResidualPickLift

CONTROLLERS = {
    "scripted": None,
    "fixed_10g": PICK_LIFT_CHECKPOINTS_DIR / "pick_lift_fixed_10g_25mm",
    "random_5_20g": PICK_LIFT_CHECKPOINTS_DIR / "pick_lift_random_5_20g_25mm",
}
POLICY_SEED = 0


@dataclass(frozen=True)
class EpisodeResult:
    """Final and transient grasp/lift evidence; contact steps use the policy rate."""

    placement_seed: int
    final_success: bool
    ever_success: bool
    ever_grasped: bool
    final_lift_m: float
    max_lift_m: float
    robot_table_contact_steps: int
    warnings: bool


@dataclass(frozen=True)
class ConditionSummary:
    """Paired-placement result for one dynamics/controller/mass combination."""

    dynamics: str
    controller: str
    mass_kg: float
    episodes: int
    success_rate: float
    mean_final_lift_cm: float
    min_final_lift_cm: float
    lost_success_episodes: int
    robot_table_contact_episodes: int
    warning_episodes: int


def make_config(mass_kg: float, calibrated: bool) -> PickLiftConfig:
    """Change only cube mass and the optional measured-stiffness file."""
    config = PickLiftConfig(mass=ObjectMassConfig(mode="fixed", fixed_kg=mass_kg))
    if calibrated:
        sim = replace(config.sim, calibration_path=resolve_project_path(DEFAULT_CALIBRATION_FILE))
        config = replace(config, sim=sim)
    return config


def summarize(dynamics: str, controller: str, mass_kg: float, episodes: list[EpisodeResult]) -> ConditionSummary:
    """Count final success independently from brief success followed by a drop."""
    if not episodes:
        raise ValueError("No completed episodes to summarize")
    lifts = np.array([ep.final_lift_m for ep in episodes]) * 100
    return ConditionSummary(
        dynamics, controller, mass_kg, len(episodes),
        float(np.mean([ep.final_success for ep in episodes])),
        float(lifts.mean()), float(lifts.min()),
        sum(ep.ever_success and not ep.final_success for ep in episodes),
        sum(ep.robot_table_contact_steps > 0 for ep in episodes),
        sum(ep.warnings for ep in episodes),
    )


def robot_touches_table(raw: SO101PickLiftEnv, bodies: frozenset[int]) -> bool:
    """Detect robot/table contact at the current control-step boundary."""
    model = raw.backend.model
    table = model.body(raw.config.sim.table_body).id
    for contact in raw.backend.data.contact:
        pair = {int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
        if table in pair and pair & bodies:
            return True
    return False


def evaluate_episode(env, policy: EpisodePolicy, seed: int) -> EpisodeResult:
    """Run existing controls; reject nonfinite state and record warning/contact evidence."""
    raw = env.unwrapped
    obs, _ = env.reset(seed=seed)
    policy.reset()
    bodies = descendant_body_ids(raw.backend.model, raw.config.sim.robot_root_body)
    ever_success = ever_grasped = warnings = False
    max_lift, table_steps = 0.0, 0
    for _ in range(raw.config.task.max_episode_steps):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError(f"Nonfinite rollout at placement seed {seed}")
        ever_success |= bool(info["is_success"])
        ever_grasped |= bool(info["grasped"])
        warnings |= raw.backend.has_warnings()
        table_steps += int(robot_touches_table(raw, bodies))
        max_lift = max(max_lift, float(info["lift_m"]))
        if terminated or truncated:
            break
    return EpisodeResult(seed, bool(info["is_success"]), ever_success, ever_grasped,
                         float(info["lift_m"]), max_lift, table_steps, warnings)


def build_condition(config: PickLiftConfig, controller: str, render_mode=None):
    """Load frozen saved policies or construct the existing scripted baseline."""
    raw = SO101PickLiftEnv(cfg=config, render_mode=render_mode)
    path = CONTROLLERS[controller]
    if path is None:
        return raw, ScriptedPickLift.for_env(raw)
    run = load_run(path)
    if run.control != "residual" or run.object != config.object:
        raw.close()
        raise ValueError("Checkpoint control mode or cube geometry does not match this comparison")
    env = ResidualPickLift(raw, ScriptedPickLift.for_env(raw))
    return env, load_policy(path / f"seed_{POLICY_SEED}", run, TrainConfig())


def condition(config: PickLiftConfig, controller: str, seeds: range) -> list[EpisodeResult]:
    """Evaluate one condition and always release simulation/normalizer resources."""
    env, policy = build_condition(config, controller)
    try:
        return [evaluate_episode(env, policy, seed) for seed in seeds]
    finally:
        env.close()
        getattr(policy, "close", lambda: None)()


def input_hashes() -> dict[str, str]:
    """Hash model, calibration, policies, normalizers, and saved run settings."""
    paths = [PickLiftConfig().sim.model_path, resolve_project_path(DEFAULT_CALIBRATION_FILE)]
    for directory in CONTROLLERS.values():
        if directory is not None:
            paths += [directory / "run_info.json"]
            paths += [directory / f"seed_{POLICY_SEED}" / name for name in ("model.zip", "vecnormalize.pkl")]
    return {str(p.relative_to(resolve_project_path("."))): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(episodes: int, output: Path, seed_offset: int) -> None:
    """Save per-episode evidence and partial progress in a new experiment directory."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    hashes = input_hashes()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "measured_stiffness_only; no reaching servo-effects or sensing wrappers",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {p: version(p) for p in ("torch", "mujoco", "gymnasium", "stable-baselines3", "numpy")},
        "seed_offset": seed_offset, "policy_seed": POLICY_SEED, "input_sha256": hashes, "conditions": [],
    }
    for calibrated in (False, True):
        dynamics = "measured_stiffness_only" if calibrated else "original"
        for controller in CONTROLLERS:
            for mass in PickLiftEvalConfig().masses_kg:
                config = make_config(mass, calibrated)
                data = condition(config, controller, range(seed_offset, seed_offset + episodes))
                summary = summarize(dynamics, controller, mass, data)
                report["conditions"].append({"summary": asdict(summary), "config": asdict(config), "episodes": [asdict(ep) for ep in data]})
                (output / "results.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
                print(f"{dynamics} {controller} {mass*1000:g} g: {summary.success_rate:.0%}; mean lift {summary.mean_final_lift_cm:.2f} cm", flush=True)
    if hashes != input_hashes():
        raise RuntimeError("Evaluation inputs changed during the run")
    print("Input hashes unchanged; evaluation complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=20, help="Placements per condition; default 20 (480 total episodes)")
    parser.add_argument("--seed-offset", type=int, default=PickLiftEvalConfig().seed_offset)
    parser.add_argument("--out", type=Path, required=True, help="New experiment directory; never overwrite existing results")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    torch.set_num_threads(1)
    run(args.episodes, args.out, args.seed_offset)

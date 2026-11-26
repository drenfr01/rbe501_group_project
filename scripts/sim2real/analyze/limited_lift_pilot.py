"""Fresh lift-only 10k SAC pilot; frozen constraints and new held-out placements."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from rbe_501_so101.config import PickLiftEvalConfig, TrainConfig
from rbe_501_so101.rl.limited_lift_training import (
    LimitedLiftPreset, LimitedLiftRepository, factory, load_policy, make_env,
)
from rbe_501_so101.rl.training import train_with_factory
from scripts.sim2real.analyze.calibrated_lift_pilot import (
    PilotConfig, ZeroResidual, artifact_hashes, decision, preflight, training_history, write,
)
from scripts.sim2real.analyze.pick_lift_calibration import evaluate_episode, input_hashes, summarize

PILOT = replace(PilotConfig(), eval_seed=68_000)


def protected_inputs() -> dict[str, str]:
    """Protect original artifacts and the failed v1 pilot; never reuse their weights."""
    result = input_hashes()
    root = Path("checkpoints/calibrated_outer_wrist_pilot")
    for relative in ("run_config.json", "seed_0/model.zip", "seed_0/vecnormalize.pkl"):
        path = root / relative
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def evaluate(preset: LimitedLiftPreset, policy, cfg: PilotConfig, name: str) -> list[dict]:
    """Report success/contact outcomes and how much residual authority was really used."""
    rows = []
    for mass in PickLiftEvalConfig().masses_kg:
        base = replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass))
        env = make_env(replace(preset, base=base))
        episodes, interventions = [], []
        try:
            for offset in range(cfg.episodes_per_mass):
                episodes.append(evaluate_episode(env, policy, cfg.eval_seed + offset))
                interventions.append(env.counters.copy())
            summary = summarize(preset.base.name, name, mass, episodes)
            rows.append({"summary": asdict(summary), "episodes": [asdict(e) for e in episodes],
                         "interventions": interventions})
            print(f"{name} {mass*1000:g}g: {summary.success_rate:.0%}, table={summary.robot_table_contact_episodes}", flush=True)
        finally:
            env.close()
    return rows


def assess(rows: list[dict]) -> dict:
    """Require the previous gate plus <=5% losses after a previously achieved lift."""
    result = decision(rows)
    holding = all(r["summary"]["lost_success_episodes"] <= .05 * r["summary"]["episodes"] for r in rows)
    return {**result, "passed_pilot_gate": result["passed_pilot_gate"] and holding,
            "holding_gate": "Each mass <=5% episodes losing previously achieved success",
            "stability_does_not_establish_learning_improvement": True}


def train(repo: LimitedLiftRepository, preset: LimitedLiftPreset, cfg: PilotConfig) -> Path:
    """Use unchanged SAC defaults in a new v2 directory with its own replay buffer."""
    training = replace(TrainConfig(), timesteps=cfg.steps, checkpoint_freq_steps=cfg.steps // 2)
    repo.create(preset, training, cfg.seed, cfg.steps)
    seed_dir = repo.root / f"seed_{cfg.seed}"
    return train_with_factory(cfg.seed, factory(repo.load(), seed_dir / "train"), algo="sac",
                              timesteps=cfg.steps, seed_dir=seed_dir, cfg=training, device="cuda")


def finish(repo: LimitedLiftRepository, output: Path, cfg: PilotConfig, seed_dir: Path) -> None:
    """Save reloaded deterministic evaluation and the complete local artifact inventory."""
    policy = load_policy(repo, cfg.seed)
    try:
        rows = evaluate(repo.load(), policy, cfg, "fresh_limited_sac")
    finally:
        policy.close()
    model = SAC.load(seed_dir / "model.zip", device="cuda")
    write(output / "learned_eval.json", rows)
    write(output / "assessment.json", assess(rows))
    write(output / "training_history.json", training_history(seed_dir))
    write(output / "artifacts.json", {"seed_dir": seed_dir, "sha256": artifact_hashes(seed_dir),
                                      "timesteps": model.num_timesteps, "device": str(model.device)})


def run(output: Path, model_dir: Path) -> None:
    """Freeze the v2 envelope before inspecting any new held-out result."""
    if model_dir.exists():
        raise FileExistsError("New model directory required")
    output.mkdir(parents=True, exist_ok=False)
    preset, hashes = LimitedLiftPreset(), protected_inputs()
    write(output / "protocol.json", {"pilot": asdict(PILOT), "preset": asdict(preset),
          "protected_sha256": hashes, "gate": assess([]), "model_dir": model_dir})
    write(output / "preflight.json", preflight(preset.base))
    env = make_env(preset)
    try:
        check_env(env, warn=True)
    finally:
        env.close()
    write(output / "scripted_eval.json", evaluate(preset, ZeroResidual(), PILOT, "scripted"))
    repo = LimitedLiftRepository(model_dir)
    seed_dir = train(repo, preset, PILOT)
    write(output / "run_config.json", json.loads(repo.path.read_text()))
    finish(repo, output, PILOT, seed_dir)
    assert hashes == protected_inputs(), "An original experiment input changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


def watch(model_dir: Path, output: Path, mass: float, seed: int, viewer: bool) -> None:
    """Replay v2 with frozen statistics, saving frames and final limiter counters."""
    from scripts.sim2real.analyze.replay_pick_lift import rollout, transition_events

    repo = LimitedLiftRepository(model_dir)
    preset = repo.load()
    fixed = replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass))
    env = make_env(replace(preset, base=fixed), "human" if viewer else None)
    policy = load_policy(repo)
    output.mkdir(parents=True, exist_ok=False)
    try:
        trace = rollout(env, policy, seed, output, frames=True, viewer=viewer)
        write(output / "trace.json", {"mass_kg": mass, "seed": seed, "preset": asdict(preset),
              "events": transition_events(trace), "trace": trace, "interventions": env.counters})
        print("Limited replay final success:", trace[-1]["success"], env.counters, flush=True)
    finally:
        env.close()
        policy.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--mass-kg", type=float, default=.01)
    parser.add_argument("--placement-seed", type=int, default=68000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.watch:
        watch(args.model_dir, args.out, args.mass_kg, args.placement_seed, args.viewer)
    else:
        run(args.out, args.model_dir)

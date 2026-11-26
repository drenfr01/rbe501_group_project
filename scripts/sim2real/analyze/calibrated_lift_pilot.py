"""Fresh 10k-transition CUDA SAC pilot with paired scripted evaluation; no legacy resume."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from rbe_501_so101.config import PickLiftEvalConfig, TrainConfig
from rbe_501_so101.rl.calibrated_pick_lift import (
    CalibratedLiftPreset, LiftPresetRepository, canonical, factory, load_policy, make_env,
)
from rbe_501_so101.rl.training import train_with_factory
from scripts.sim2real.analyze.pick_lift_calibration import evaluate_episode, input_hashes, summarize


@dataclass(frozen=True)
class PilotConfig:
    """A deliberately short one-seed run, with separate frozen evaluation placements."""

    steps: int = 10_000
    seed: int = 0
    eval_seed: int = 66_000
    episodes_per_mass: int = 20


class ZeroResidual:
    """Evaluate the scripted controller through exactly the training wrapper stack."""

    def reset(self) -> None:
        """No policy state; the environment resets the scripted controller."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """A zero residual leaves the scripted action intact."""
        return np.zeros(6, dtype=np.float32)


def write(path: Path, value: object) -> None:
    """Persist pilot evidence without touching earlier experiment directories."""
    path.write_text(json.dumps(canonical(value), indent=2), encoding="utf-8")


def evaluate(preset: CalibratedLiftPreset, policy, cfg: PilotConfig, name: str) -> list[dict]:
    """Pair four fixed masses and held-out placements between script and learned residual."""
    results = []
    for mass in PickLiftEvalConfig().masses_kg:
        fixed = replace(preset, mass=replace(preset.mass, mode="fixed", fixed_kg=mass))
        env = make_env(fixed)
        try:
            episodes = [evaluate_episode(env, policy, cfg.eval_seed + n) for n in range(cfg.episodes_per_mass)]
            summary = summarize(preset.name, name, mass, episodes)
            results.append({"summary": asdict(summary), "episodes": [asdict(e) for e in episodes]})
            print(f"{name} {mass*1000:g}g: {summary.success_rate:.0%}, table={summary.robot_table_contact_episodes}", flush=True)
        finally:
            env.close()
    return results


def preflight(preset: CalibratedLiftPreset) -> dict:
    """Verify CUDA arithmetic and the actual residual environment before training."""
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot requires CUDA; no silent CPU fallback")
    result = (torch.ones(32, device="cuda") * 2).sum().item()
    assert result == 64
    env = make_env(preset)
    try:
        check_env(env, warn=True)
    finally:
        env.close()
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "versions": {p: version(p) for p in ("mujoco", "gymnasium", "stable-baselines3", "numpy")}}


def training_history(seed_dir: Path) -> list[dict]:
    """Export rewards and contact evidence from SB3's complete episode monitor."""
    with (seed_dir / "train.monitor.csv").open() as stream:
        next(stream)
        return list(csv.DictReader(stream))


def artifact_hashes(seed_dir: Path) -> dict[str, str]:
    """Identify exact local pilot model, statistics and resumable checkpoint files."""
    return {str(p.relative_to(seed_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in seed_dir.rglob("*") if p.is_file()}


def decision(results: list[dict]) -> dict:
    """Use the frozen gate; a short pilot never authorizes automatic full training."""
    passed = len(results) == len(PickLiftEvalConfig().masses_kg) and all(r["summary"]["success_rate"] >= .95 and
                 r["summary"]["robot_table_contact_episodes"] <= .05 * r["summary"]["episodes"] and
                 r["summary"]["warning_episodes"] == 0 for r in results)
    return {"passed_pilot_gate": passed, "start_full_training": False,
            "gate": "Each mass >=95% success, <=5% sampled table-contact episodes, no warnings"}


def train(repo: LiftPresetRepository, preset: CalibratedLiftPreset, cfg: PilotConfig) -> Path:
    """Reuse the repository's SAC defaults and checkpoint code in a fresh directory."""
    training = replace(TrainConfig(), timesteps=cfg.steps, checkpoint_freq_steps=cfg.steps // 2)
    repo.create(preset, training, cfg.seed, cfg.steps)
    seed_dir = repo.root / f"seed_{cfg.seed}"
    return train_with_factory(cfg.seed, factory(repo.load(), seed_dir / "train"), algo="sac",
                              timesteps=cfg.steps, seed_dir=seed_dir, cfg=training, device="cuda")


def run(output: Path, model_dir: Path) -> None:
    """Freeze inputs and evaluation criteria before a short fresh-policy experiment."""
    if model_dir.exists():
        raise FileExistsError("Pilot model directory must be new")
    output.mkdir(parents=True, exist_ok=False)
    cfg, preset = PilotConfig(), CalibratedLiftPreset()
    hashes = input_hashes()
    write(output / "protocol.json", {"pilot": asdict(cfg), "preset": asdict(preset),
          "gate": decision([])["gate"], "model_dir": model_dir, "original_sha256": hashes})
    write(output / "preflight.json", preflight(preset))
    write(output / "scripted_eval.json", evaluate(preset, ZeroResidual(), cfg, "scripted"))
    repo = LiftPresetRepository(model_dir)
    seed_dir = train(repo, preset, cfg)
    write(output / "run_config.json", json.loads(repo.path.read_text()))
    finish(output, repo, cfg, seed_dir)
    assert hashes == input_hashes(), "An original model, calibration or checkpoint changed"


def finish(output: Path, repo: LiftPresetRepository, cfg: PilotConfig, seed_dir: Path) -> None:
    """Evaluate the saved/reloaded policy, export history and preserve provenance."""
    policy = load_policy(repo, cfg.seed)
    try:
        results = evaluate(repo.load(), policy, cfg, "fresh_sac")
    finally:
        policy.close()
    model = SAC.load(seed_dir / "model.zip", device="cuda")
    write(output / "learned_eval.json", results)
    write(output / "assessment.json", decision(results))
    write(output / "training_history.json", training_history(seed_dir))
    write(output / "artifacts.json", {"seed_dir": seed_dir, "sha256": artifact_hashes(seed_dir),
          "timesteps": model.num_timesteps, "device": str(model.device), "learning_starts": model.learning_starts,
          "batch_size": model.batch_size, "buffer_size": model.buffer_size, "gradient_steps": model.gradient_steps,
          "train_freq": str(model.train_freq), "policy": str(model.policy)})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write(output / "provenance.json", {"base_commit": commit, "original_inputs_unchanged": True,
          "note": "New source files are recorded in the subsequent pilot commit. Local checkpoints are excluded by .gitignore."})


def watch(model_dir: Path, output: Path, mass: float, seed: int, viewer: bool, frames: bool) -> None:
    """Replay the saved pilot using its own preset and frozen observation statistics."""
    from scripts.sim2real.analyze.replay_pick_lift import rollout, transition_events

    repo = LiftPresetRepository(model_dir)
    preset = repo.load()
    fixed = replace(preset, mass=replace(preset.mass, mode="fixed", fixed_kg=mass))
    output.mkdir(parents=True, exist_ok=False)
    env, policy = make_env(fixed, "human" if viewer else None), load_policy(repo)
    try:
        trace = rollout(env, policy, seed, output, frames, viewer)
        write(output / "trace.json", {"preset": asdict(preset), "mass_kg": mass,
              "placement_seed": seed, "events": transition_events(trace), "trace": trace})
        print(f"Pilot replay: success={trace[-1]['success']}, lift={trace[-1]['lift_m']:.4f} m")
    finally:
        env.close()
        policy.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true", help="Replay saved pilot; never train")
    parser.add_argument("--mass-kg", type=float, default=.01)
    parser.add_argument("--placement-seed", type=int, default=66000)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--frames", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.watch:
        watch(args.model_dir, args.out, args.mass_kg, args.placement_seed, args.viewer, args.frames)
    else:
        run(args.out, args.model_dir)

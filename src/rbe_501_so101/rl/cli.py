"""Command-line entry point for reach training and evaluation."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rbe_501_so101.config import PRESETS, TrainConfig
from rbe_501_so101.rl.checkpoints import LEGACY_PRESET, CheckpointRepository
from rbe_501_so101.rl.evaluation import run_evaluation
from rbe_501_so101.rl.training import ALGORITHMS, resolve_device, train_seed

AlgoName = Literal["sac", "ppo"]


@dataclass(frozen=True)
class RunSettings:
    """CLI options with defaults taken from TrainConfig."""

    algo: AlgoName
    timesteps: int
    seeds: tuple[int, ...]
    randomize_dynamics: bool
    evaluate: bool
    model_dir: Path
    num_episodes: int
    render: bool
    cpu: bool
    preset: str


def parse_args(cfg: TrainConfig, argv: list[str] | None = None) -> RunSettings:
    """Command-line interface with defaults taken from TrainConfig."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default=cfg.algo)
    parser.add_argument("--timesteps", type=int, default=cfg.timesteps)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.seeds))
    parser.add_argument("--randomize-dynamics", action="store_true", help="randomize dynamics during training")
    parser.add_argument("--eval", action="store_true", help="evaluate saved checkpoints instead of training")
    parser.add_argument("--model-dir", type=Path, default=cfg.model_dir)
    parser.add_argument("--num-episodes", type=int, default=cfg.eval_episodes)
    parser.add_argument("--render", action="store_true", help="open the viewer during evaluation")
    parser.add_argument("--cpu", action="store_true", help="train on CPU even if a GPU is present")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=LEGACY_PRESET,
        help="training variant (evaluation reads each checkpoint's recorded preset)",
    )
    args = parser.parse_args(argv)
    return RunSettings(
        algo=args.algo,
        timesteps=args.timesteps,
        seeds=tuple(args.seeds),
        randomize_dynamics=args.randomize_dynamics,
        evaluate=args.eval,
        model_dir=args.model_dir,
        num_episodes=args.num_episodes,
        render=args.render,
        cpu=args.cpu,
        preset=args.preset,
    )


def main(argv: list[str] | None = None) -> None:
    """Train each seed, or evaluate saved checkpoints against baselines."""
    cfg = TrainConfig()
    settings = parse_args(cfg, argv)
    if settings.evaluate:
        print(json.dumps(_evaluation_report(settings, cfg), indent=2))
        return
    device = resolve_device(prefer_gpu=not settings.cpu)
    repo = CheckpointRepository(settings.model_dir, cfg)
    for seed in settings.seeds:
        repo.ensure_untrained(seed)
    for seed in settings.seeds:
        path = train_seed(
            seed,
            algo=settings.algo,
            timesteps=settings.timesteps,
            randomize_dynamics=settings.randomize_dynamics,
            repo=repo,
            preset_name=settings.preset,
            cfg=cfg,
            device=device,
        )
        print(f"saved {path}")


def _evaluation_report(settings: RunSettings, cfg: TrainConfig) -> dict:
    """Run the full evaluation using the parsed CLI settings."""
    return run_evaluation(
        settings.seeds,
        settings.algo,
        settings.model_dir,
        settings.num_episodes,
        cfg,
        settings.render,
    )

"""Write a training run's config, eval report, logs, and plots to an experiment folder.

HOW TO RUN (from the project folder):
    python scripts/report_experiment.py --name 2026-10-11_sac_scaled_reward \
        --model-dir checkpoints/reach_scaled --eval-report logs/eval_report_scaled.raw \
        --baseline-report logs/eval_report.json

Produces experiments/<name>/ with config.json, eval_report.json, per-seed
monitor CSVs, training_reward.png, and holding_error.png.
"""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from rbe_501_so101.config import (
    PROJECT_ROOT,
    DynamicsRandomizationConfig,
    ReachTaskConfig,
    RewardConfig,
    SimConfig,
    TargetSamplerConfig,
    TrainConfig,
)
from rbe_501_so101.rl.plots import plot_holding_error, plot_training_reward
from rbe_501_so101.rl.reporting import load_curves, load_report

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MONITOR_NAME = "train.monitor.csv"


def write_config(out_dir: Path) -> None:
    """Snapshot every config used by training and evaluation."""
    configs = {
        "train": TrainConfig(),
        "reward": RewardConfig(),
        "task": ReachTaskConfig(),
        "sampler": TargetSamplerConfig(),
        "sim": SimConfig(),
        "randomization": DynamicsRandomizationConfig(),
    }
    payload = {name: asdict(cfg) for name, cfg in configs.items()}
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))


def copy_monitors(model_dir: Path, seeds: tuple[int, ...], out_dir: Path) -> None:
    """Copy each seed's Monitor CSV (renamed so `*.monitor.csv` gitignore rules skip it)."""
    for seed in seeds:
        shutil.copy(model_dir / f"seed_{seed}" / MONITOR_NAME, out_dir / f"seed_{seed}_monitor.csv")


def holding_rows(report: dict, baseline: dict | None) -> dict[str, dict]:
    """Policies to compare in the holding-error chart, in display order."""
    rows: dict[str, dict] = {}
    if baseline is not None:
        rows["previous reward (nominal)"] = baseline["trained_nominal"]
    rows["this run (nominal)"] = report["trained_nominal"]
    rows["this run (randomized)"] = report["trained_randomized_dynamics"]
    rows["IK hold"] = report["baselines"]["ik_hold"]
    return rows


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="experiment folder name under experiments/")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--eval-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, help="earlier eval report to compare against")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TrainConfig().seeds))
    return parser.parse_args()


def main() -> None:
    """Assemble the experiment folder."""
    args = parse_args()
    seeds = tuple(args.seeds)
    out_dir = EXPERIMENTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    report = load_report(args.eval_report)
    baseline = load_report(args.baseline_report) if args.baseline_report else None
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    write_config(out_dir)
    copy_monitors(args.model_dir, seeds, out_dir)
    plot_training_reward(load_curves(args.model_dir, seeds, MONITOR_NAME), out_dir / "training_reward.png")
    plot_holding_error(holding_rows(report, baseline), TrainConfig(), out_dir / "holding_error.png")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

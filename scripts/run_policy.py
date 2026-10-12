"""Watch a trained reach policy in the MuJoCo viewer at real-time speed.

HOW TO RUN (from the project folder):
    python scripts/run_policy.py --model-dir checkpoints/reach_scaled --seed 0
    python scripts/run_policy.py --model-dir checkpoints/reach_scaled --seed 2 --episodes 10
    python scripts/run_policy.py --model-dir checkpoints/reach_scaled --headless

Each episode samples a new target (red sphere), runs the deterministic policy
for 150 steps, and prints the final and holding-window distance. Close the
viewer window to stop early.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from rbe_501_so101.config import TrainConfig
from rbe_501_so101.envs import SO101ReachEnv
from rbe_501_so101.rl.checkpoints import CheckpointRepository
from rbe_501_so101.rl.evaluation import load_trained_policy, make_eval_env
from rbe_501_so101.rl.policies import TrainedPolicy
from rbe_501_so101.rl.training import ALGORITHMS

M_TO_MM = 1000.0


def play_episode(env: SO101ReachEnv, policy: TrainedPolicy, seed: int, realtime: bool) -> list[float] | None:
    """Run one episode; return per-step distances, or None if the viewer was closed."""
    obs, _ = env.reset(seed=seed)
    distances, truncated = [], False
    while not truncated:
        if not env.viewer_open:
            return None
        tick = time.perf_counter()
        obs, _, _, truncated, info = env.step(policy(obs))
        distances.append(info["distance_m"])
        if realtime:
            time.sleep(max(0.0, env.control_dt - (time.perf_counter() - tick)))
    return distances


def parse_args(cfg: TrainConfig) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=cfg.model_dir)
    parser.add_argument("--seed", type=int, default=0, help="which trained seed to load")
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default=cfg.algo)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--target-seed", type=int, default=cfg.eval_seed_offset, help="seed of the first target")
    parser.add_argument("--randomize-dynamics", action="store_true")
    parser.add_argument("--headless", action="store_true", help="no viewer; run as fast as possible")
    return parser.parse_args()


def main() -> None:
    """Load one checkpoint and play episodes."""
    cfg = TrainConfig()
    args = parse_args(cfg)
    repo = CheckpointRepository(args.model_dir, cfg)
    policy = load_trained_policy(repo, args.seed, args.algo)
    render_mode = None if args.headless else "human"
    env = make_eval_env(repo.load_preset(args.seed), args.randomize_dynamics, render_mode)
    try:
        for episode in range(args.episodes):
            distances = play_episode(env, policy, args.target_seed + episode, realtime=not args.headless)
            if distances is None:
                break
            hold = M_TO_MM * float(np.mean(distances[-cfg.holding_window_steps :]))
            print(f"episode {episode}: final {M_TO_MM * distances[-1]:.1f} mm, holding mean {hold:.1f} mm")
    finally:
        env.close()
        policy.close()


if __name__ == "__main__":
    main()

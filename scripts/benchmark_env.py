"""Benchmark SO101ReachEnv throughput, or run the viewer smoke test.

HOW TO RUN (from the project folder):
    python scripts/benchmark_env.py              # headless throughput
    python scripts/benchmark_env.py --render     # 3-episode viewer smoke test

Throughput is a performance benchmark, not a correctness test: it reports
policy steps per second over random actions with auto-reset. The smoke test
plays random-action episodes at real-time speed so you can confirm the arm is
mounted on the table and sags smoothly, the red target sphere moves to each
new target above the tabletop, and closing the window exits cleanly.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass

from rbe_501_so101.config import BenchmarkConfig, SimConfig
from rbe_501_so101.envs import SO101ReachEnv


@dataclass(frozen=True)
class ThroughputReport:
    """Headless throughput measurement."""

    steps: int
    seconds: float
    steps_per_s: float
    target_steps_per_s: float
    meets_target: bool


def measure_throughput(cfg: BenchmarkConfig) -> ThroughputReport:
    """Run random actions headlessly and time them."""
    env = SO101ReachEnv()
    env.action_space.seed(cfg.seed)
    env.reset(seed=cfg.seed)
    start = time.perf_counter()
    for _ in range(cfg.throughput_steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    env.close()
    rate = cfg.throughput_steps / elapsed
    target = cfg.throughput_target_steps_per_s
    return ThroughputReport(cfg.throughput_steps, elapsed, rate, target, rate >= target)


def run_viewer_smoke_test(cfg: BenchmarkConfig) -> None:
    """Play random-action episodes in the passive viewer at real-time speed."""
    env = SO101ReachEnv(render_mode="human")
    period = SimConfig().n_substeps * env.backend.model.opt.timestep
    env.action_space.seed(cfg.seed)
    try:
        for episode in range(cfg.smoke_episodes):
            _, info = env.reset(seed=cfg.seed + episode)
            print(f"episode {episode}: target {info['target'].round(3).tolist()}")
            if not _play_episode(env, period):
                break
    finally:
        env.close()


def _play_episode(env: SO101ReachEnv, period_s: float) -> bool:
    """Step until truncation; return False if the viewer window was closed."""
    truncated = False
    while not truncated:
        if not env.viewer_open:
            return False
        tick = time.perf_counter()
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        time.sleep(max(0.0, period_s - (time.perf_counter() - tick)))
    return True


def main() -> None:
    """Parse arguments and run the requested benchmark."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", action="store_true", help="run the viewer smoke test")
    args = parser.parse_args()
    cfg = BenchmarkConfig()
    if args.render:
        run_viewer_smoke_test(cfg)
    else:
        print(json.dumps(asdict(measure_throughput(cfg)), indent=2))


if __name__ == "__main__":
    main()

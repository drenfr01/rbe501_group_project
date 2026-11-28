"""Run a scripted or trained pick-and-lift policy."""

import argparse
import time
from pathlib import Path

import gymnasium as gym

from rbe_501_so101.config import ObjectMassConfig, PickLiftEvalConfig, TrainConfig
from rbe_501_so101.control import ScriptedPickLift
from rbe_501_so101.rl.pick_lift import EpisodePolicy, load_policy, load_run, make_pick_lift_env

M_TO_CM = 100.0


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=None, help="trained run; omit for the scripted controller")
    parser.add_argument("--seed", type=int, default=0, help="which trained seed to load")
    parser.add_argument("--mass-kg", type=float, default=ObjectMassConfig().fixed_kg)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--placement-seed", type=int, default=PickLiftEvalConfig().seed_offset)
    parser.add_argument("--headless", action="store_true", help="no viewer; run as fast as possible")
    return parser.parse_args()


def build(args: argparse.Namespace) -> tuple[gym.Env, EpisodePolicy]:
    """The env (residual-wrapped if the run used it) and the policy to drive it."""
    mass = ObjectMassConfig(mode="fixed", fixed_kg=args.mass_kg)
    render_mode = None if args.headless else "human"
    if args.model_dir is None:
        env = make_pick_lift_env(mass, "direct", render_mode)
        return env, ScriptedPickLift.for_env(env)
    run = load_run(args.model_dir)
    env = make_pick_lift_env(mass, run.control, render_mode, object_cfg=run.object)
    return env, load_policy(args.model_dir / f"seed_{args.seed}", run, TrainConfig())


def play_episode(env: gym.Env, policy: EpisodePolicy, seed: int, realtime: bool) -> dict | None:
    """Run one episode; return the last info, or None if the viewer was closed."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    truncated, info = False, {}
    while not truncated:
        if not env.unwrapped.viewer_open:
            return None
        tick = time.perf_counter()
        obs, _, _, truncated, info = env.step(policy(obs))
        if realtime:
            time.sleep(max(0.0, env.unwrapped.control_dt - (time.perf_counter() - tick)))
    return info


def main() -> None:
    """Play episodes and report the outcome of each."""
    args = parse_args()
    env, policy = build(args)
    try:
        for episode in range(args.episodes):
            info = play_episode(env, policy, args.placement_seed + episode, realtime=not args.headless)
            if info is None:
                break
            outcome = "SUCCESS" if info["is_success"] else "fail"
            print(f"episode {episode}: {outcome}, lift {M_TO_CM * info['lift_m']:.1f} cm, grasped {info['grasped']}")
    finally:
        env.close()
        getattr(policy, "close", lambda: None)()


if __name__ == "__main__":
    main()

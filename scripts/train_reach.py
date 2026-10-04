"""Train, evaluate, and benchmark the SO-101 reaching policy.

HOW TO RUN (from the project folder):
    python scripts/train_reach.py --algo sac --timesteps 300000 --seeds 0 1 2
    python scripts/train_reach.py --eval --model-dir checkpoints/reach/ --num-episodes 20
    python scripts/train_reach.py --eval --render

Training wraps SO101Reach-v0 in Monitor + VecNormalize (observations
normalized, rewards left raw) and saves the model and normalization statistics
to <model-dir>/seed_<n>/. Evaluation measures the gripper-to-target error over
the final holding window of each episode for the trained policy (nominal and
randomized dynamics) and for a random policy and an IK-hold baseline on the
same targets, then prints a JSON report. The run passes if the nominal mean is
under 5 mm, the p95 under 10 mm, every seed meets the mean threshold, the policy
beats IK-hold, and the arm holds still.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rbe_501_so101.config import IKConfig, ReachTaskConfig, SimConfig, TrainConfig
from rbe_501_so101.envs import REACH_ENV_ID, SO101ReachEnv
from rbe_501_so101.kinematics import PositionIKSolver

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {"sac": SAC, "ppo": PPO}
N_ARM = len(SimConfig().arm_joints)
QPOS_SLICE = slice(0, N_ARM)
QVEL_SLICE = slice(N_ARM, 2 * N_ARM)
CMD_ERR_SLICE = slice(2 * N_ARM, 3 * N_ARM)
M_TO_MM = 1000.0
P95 = 95


# ---------------------------------------------------------------- training


def make_env_factory(randomize: bool, monitor_path: Path | None = None) -> Callable[[], gym.Env]:
    """Return a thunk creating a (monitored) reach env for vectorization."""

    def factory() -> gym.Env:
        env = gym.make(REACH_ENV_ID, randomize_dynamics=randomize)
        return Monitor(env, filename=str(monitor_path) if monitor_path else None)

    return factory


def build_model(algo: str, env: VecNormalize, seed: int, cfg: TrainConfig) -> BaseAlgorithm:
    """Construct SAC (default) or PPO with the configured hyperparameters."""
    if algo == "sac":
        return SAC("MlpPolicy", env, learning_rate=cfg.learning_rate, buffer_size=cfg.buffer_size,
                   batch_size=cfg.batch_size, gamma=cfg.gamma, seed=seed, verbose=1)
    return PPO("MlpPolicy", env, learning_rate=cfg.learning_rate, gamma=cfg.gamma, seed=seed, verbose=1)


def train_seed(seed: int, args: argparse.Namespace, cfg: TrainConfig) -> Path:
    """Train one seed and save the model plus normalization statistics."""
    seed_dir = args.model_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    venv = DummyVecEnv([make_env_factory(args.randomize_dynamics, seed_dir / "train")])
    venv.seed(seed)
    env = VecNormalize(venv, norm_obs=True, norm_reward=False, gamma=cfg.gamma)
    model = build_model(args.algo, env, seed, cfg)
    model.learn(total_timesteps=args.timesteps)
    model.save(seed_dir / cfg.model_filename)
    env.save(seed_dir / cfg.vecnormalize_filename)
    env.close()
    return seed_dir


# ---------------------------------------------------------------- policies


class EvalPolicy(Protocol):
    """A policy evaluated on raw (unnormalized) observations."""

    def reset(self, env: SO101ReachEnv) -> None:
        """Prepare for a new episode (after env.reset)."""
        ...

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return an action in [-1, 1]."""
        ...


class TrainedPolicy:
    """SB3 model behind saved VecNormalize observation statistics."""

    def __init__(self, model: BaseAlgorithm, normalizer: VecNormalize) -> None:
        self._model = model
        self._normalizer = normalizer

    def reset(self, env: SO101ReachEnv) -> None:
        """Stateless policy; nothing to reset."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action for the normalized observation."""
        action, _ = self._model.predict(self._normalizer.normalize_obs(obs), deterministic=True)
        return action


class RandomPolicy:
    """Uniformly random actions."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, env: SO101ReachEnv) -> None:
        """Stateless policy; nothing to reset."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Random action in [-1, 1]."""
        return self._rng.uniform(-1.0, 1.0, size=N_ARM).astype(np.float32)


class IKHoldPolicy:
    """Solve IK for the target, drive the command to that pose, never correct sag."""

    def __init__(self, solver: PositionIKSolver, task: ReachTaskConfig) -> None:
        self._solver = solver
        self._delta_max = task.delta_q_max_rad
        self._q_goal = np.zeros(N_ARM)

    def reset(self, env: SO101ReachEnv) -> None:
        """Solve IK from the current commanded pose for the new target."""
        self._q_goal = self._solver.solve(env.target, env.state.q_cmd).q[:N_ARM]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Rate-limited step of the command toward the IK pose."""
        q_cmd = obs[QPOS_SLICE] + obs[CMD_ERR_SLICE]
        return np.clip((self._q_goal - q_cmd) / self._delta_max, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class EpisodeStats:
    """Holding-window errors (m) and mean arm joint speed (rad/s) for one episode."""

    holding_errors_m: np.ndarray
    mean_joint_speed: float


@dataclass(frozen=True)
class PolicyReport:
    """Aggregate holding-window statistics for one policy."""

    mean_mm: float
    p95_mm: float
    per_seed_mean_mm: dict[str, float]
    mean_joint_speed_rad_s: float


def run_episode(env: SO101ReachEnv, policy: EvalPolicy, seed: int, window: int) -> EpisodeStats:
    """Roll out one episode and keep the final `window` steps."""
    obs, _ = env.reset(seed=seed)
    policy.reset(env)
    errors, speeds, truncated = [], [], False
    while not truncated and env.viewer_open:
        obs, _, _, truncated, info = env.step(policy(obs))
        errors.append(info["distance_m"])
        speeds.append(float(np.mean(np.abs(obs[QVEL_SLICE]))))
    return EpisodeStats(np.array(errors[-window:]), float(np.mean(speeds[-window:])))


def evaluate(env: SO101ReachEnv, policy: EvalPolicy, args: argparse.Namespace, cfg: TrainConfig) -> list[EpisodeStats]:
    """Evaluate a policy on the shared sequence of evaluation targets."""
    seeds = [cfg.eval_seed_offset + ep for ep in range(args.num_episodes)]
    return [run_episode(env, policy, s, cfg.holding_window_steps) for s in seeds]


def summarize(per_seed: dict[str, list[EpisodeStats]]) -> PolicyReport:
    """Mean, p95, per-seed mean (mm), and mean holding joint speed."""
    stats = [ep for episodes in per_seed.values() for ep in episodes]
    errors = np.concatenate([ep.holding_errors_m for ep in stats])
    per_seed_mean = {k: M_TO_MM * float(np.mean([ep.holding_errors_m for ep in v])) for k, v in per_seed.items()}
    return PolicyReport(
        mean_mm=M_TO_MM * float(np.mean(errors)),
        p95_mm=M_TO_MM * float(np.percentile(errors, P95)),
        per_seed_mean_mm=per_seed_mean,
        mean_joint_speed_rad_s=float(np.mean([ep.mean_joint_speed for ep in stats])),
    )


def load_trained_policy(seed_dir: Path, algo: str, cfg: TrainConfig) -> TrainedPolicy:
    """Load a checkpoint and its frozen normalization statistics."""
    normalizer = VecNormalize.load(str(seed_dir / cfg.vecnormalize_filename), DummyVecEnv([SO101ReachEnv]))
    normalizer.training = False
    model = ALGORITHMS[algo].load(seed_dir / cfg.model_filename)
    return TrainedPolicy(model, normalizer)


def evaluate_trained(args: argparse.Namespace, cfg: TrainConfig, randomize: bool) -> PolicyReport:
    """Evaluate every seed's checkpoint on nominal or randomized dynamics."""
    render = "human" if args.render and not randomize else None
    env = SO101ReachEnv(render_mode=render, randomize_dynamics=randomize)
    per_seed = {}
    for seed in args.seeds:
        policy = load_trained_policy(args.model_dir / f"seed_{seed}", args.algo, cfg)
        per_seed[str(seed)] = evaluate(env, policy, args, cfg)
    env.close()
    return summarize(per_seed)


def evaluate_baselines(args: argparse.Namespace, cfg: TrainConfig) -> dict[str, PolicyReport]:
    """Random and IK-hold baselines on the same targets as the trained policy."""
    env = SO101ReachEnv()
    solver = PositionIKSolver.from_config(SimConfig(), IKConfig())
    baselines: dict[str, EvalPolicy] = {
        "random": RandomPolicy(cfg.eval_seed_offset),
        "ik_hold": IKHoldPolicy(solver, ReachTaskConfig()),
    }
    reports = {name: summarize({"all": evaluate(env, policy, args, cfg)}) for name, policy in baselines.items()}
    env.close()
    return reports


def passes(trained: PolicyReport, ik_hold: PolicyReport, cfg: TrainConfig) -> bool:
    """Apply the milestone success criteria to the nominal-dynamics report."""
    mean_mm, p95_mm = M_TO_MM * cfg.success_mean_m, M_TO_MM * cfg.success_p95_m
    return (
        trained.mean_mm < mean_mm
        and trained.p95_mm < p95_mm
        and all(v < mean_mm for v in trained.per_seed_mean_mm.values())
        and trained.mean_mm < ik_hold.mean_mm
        and trained.mean_joint_speed_rad_s < cfg.holding_joint_speed_max_rad_s
    )


def run_evaluation(args: argparse.Namespace, cfg: TrainConfig) -> dict:
    """Evaluate trained policies and baselines and assemble the JSON report."""
    nominal = evaluate_trained(args, cfg, randomize=False)
    randomized = evaluate_trained(args, cfg, randomize=True)
    baselines = evaluate_baselines(args, cfg)
    return {
        "trained_nominal": asdict(nominal),
        "trained_randomized_dynamics": asdict(randomized),
        "baselines": {name: asdict(report) for name, report in baselines.items()},
        "holding_window_steps": cfg.holding_window_steps,
        "episodes_per_seed": args.num_episodes,
        "passed": passes(nominal, baselines["ik_hold"], cfg),
    }


# ---------------------------------------------------------------- CLI


def parse_args(cfg: TrainConfig) -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    """Train each seed, or evaluate saved checkpoints against baselines."""
    cfg = TrainConfig()
    args = parse_args(cfg)
    if args.eval:
        print(json.dumps(run_evaluation(args, cfg), indent=2))
        return
    for seed in args.seeds:
        print(f"saved {train_seed(seed, args, cfg)}")


if __name__ == "__main__":
    main()

"""Evaluate trained reach policies against random and IK-hold baselines."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rbe_501_so101.config import IKConfig, ReachTaskConfig, SimConfig, TrainConfig
from rbe_501_so101.envs import SO101ReachEnv
from rbe_501_so101.envs.types import ObservationLayout
from rbe_501_so101.kinematics import PositionIKSolver
from rbe_501_so101.rl.policies import EvalPolicy, IKHoldPolicy, RandomPolicy, TrainedPolicy
from rbe_501_so101.rl.training import ALGORITHMS, make_env_factory

M_TO_MM = 1000.0
P95 = 95
AlgoName = Literal["sac", "ppo"]


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


def run_episode(env: SO101ReachEnv, policy: EvalPolicy, seed: int, window: int) -> EpisodeStats | None:
    """Roll out one episode. Return None if the viewer closed before truncation."""
    obs, _ = env.reset(seed=seed)
    policy.reset(env.target, env.state)
    errors, speeds, truncated = [], [], False
    while not truncated:
        if not env.viewer_open:
            return None
        obs, _, _, truncated, info = env.step(policy(obs))
        errors.append(info["distance_m"])
        speeds.append(float(np.mean(np.abs(obs[env.layout.qvel]))))
    return EpisodeStats(np.array(errors[-window:]), float(np.mean(speeds[-window:])))


def completed_episodes(stats: list[EpisodeStats | None]) -> list[EpisodeStats]:
    """Drop incomplete (viewer-closed) episodes; require at least one finished."""
    done = [ep for ep in stats if ep is not None]
    if not done:
        raise RuntimeError("No complete evaluation episodes")
    return done


def evaluate(env: SO101ReachEnv, policy: EvalPolicy, num_episodes: int, cfg: TrainConfig) -> list[EpisodeStats]:
    """Evaluate a policy on the shared sequence of evaluation targets."""
    seeds = [cfg.eval_seed_offset + ep for ep in range(num_episodes)]
    return completed_episodes([run_episode(env, policy, s, cfg.holding_window_steps) for s in seeds])


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


def load_trained_policy(seed_dir: Path, algo: AlgoName, cfg: TrainConfig) -> TrainedPolicy:
    """Load a checkpoint and its frozen normalization statistics."""
    venv = DummyVecEnv([make_env_factory(False)])
    normalizer = VecNormalize.load(str(seed_dir / cfg.vecnormalize_filename), venv)
    normalizer.training = False
    model = ALGORITHMS[algo].load(seed_dir / cfg.model_filename)
    return TrainedPolicy(model, normalizer, venv)


def evaluate_trained(
    seeds: tuple[int, ...],
    algo: AlgoName,
    model_dir: Path,
    num_episodes: int,
    cfg: TrainConfig,
    randomize: bool,
    render: bool,
) -> PolicyReport:
    """Evaluate every seed's checkpoint on nominal or randomized dynamics."""
    env = SO101ReachEnv(render_mode="human" if render and not randomize else None, randomize_dynamics=randomize)
    per_seed: dict[str, list[EpisodeStats]] = {}
    policies: list[TrainedPolicy] = []
    try:
        for seed in seeds:
            policy = load_trained_policy(model_dir / f"seed_{seed}", algo, cfg)
            policies.append(policy)
            per_seed[str(seed)] = evaluate(env, policy, num_episodes, cfg)
        return summarize(per_seed)
    finally:
        env.close()
        for policy in policies:
            policy.close()


def evaluate_baselines(num_episodes: int, cfg: TrainConfig) -> dict[str, PolicyReport]:
    """Random and IK-hold baselines on the same targets as the trained policy."""
    sim_cfg = SimConfig()
    env = SO101ReachEnv()
    layout = ObservationLayout(n_arm=len(sim_cfg.arm_joints))
    solver = PositionIKSolver.from_config(sim_cfg, IKConfig())
    baselines: dict[str, EvalPolicy] = {
        "random": RandomPolicy(cfg.eval_seed_offset, layout.n_arm),
        "ik_hold": IKHoldPolicy(solver, ReachTaskConfig(), layout),
    }
    try:
        return {name: summarize({"all": evaluate(env, policy, num_episodes, cfg)}) for name, policy in baselines.items()}
    finally:
        env.close()


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


def run_evaluation(
    seeds: tuple[int, ...],
    algo: AlgoName,
    model_dir: Path,
    num_episodes: int,
    cfg: TrainConfig,
    render: bool,
) -> dict:
    """Evaluate trained policies and baselines and assemble the JSON report."""
    nominal = evaluate_trained(seeds, algo, model_dir, num_episodes, cfg, randomize=False, render=render)
    randomized = evaluate_trained(seeds, algo, model_dir, num_episodes, cfg, randomize=True, render=False)
    baselines = evaluate_baselines(num_episodes, cfg)
    return {
        "trained_nominal": asdict(nominal),
        "trained_randomized_dynamics": asdict(randomized),
        "baselines": {name: asdict(report) for name, report in baselines.items()},
        "holding_window_steps": cfg.holding_window_steps,
        "episodes_per_seed": num_episodes,
        "passed": passes(nominal, baselines["ik_hold"], cfg),
    }

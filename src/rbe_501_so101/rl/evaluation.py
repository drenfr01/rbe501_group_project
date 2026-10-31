"""Evaluate trained reach policies against random and IK-hold baselines."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rbe_501_so101.config import (
    PRESETS,
    EnvOptions,
    ExperimentPreset,
    IKConfig,
    PerturbationConfig,
    ReachTaskConfig,
    RobustnessConfig,
    SimConfig,
    TrainConfig,
)
from rbe_501_so101.envs import SO101ReachEnv
from rbe_501_so101.envs.factory import make_reach_env
from rbe_501_so101.envs.mujoco_backend import RenderMode
from rbe_501_so101.envs.types import ObservationLayout
from rbe_501_so101.kinematics import PositionIKSolver
from rbe_501_so101.rl.checkpoints import CheckpointRepository
from rbe_501_so101.rl.policies import EvalPolicy, IKHoldPolicy, RandomPolicy, TrainedPolicy
from rbe_501_so101.rl.training import ALGORITHMS, make_env_factory

M_TO_MM = 1000.0
P95 = 95
AlgoName = Literal["sac", "ppo"]


@dataclass(frozen=True)
class EpisodeStats:
    """Holding-window errors (m), mean arm joint speed (rad/s), and actions for one episode."""

    holding_errors_m: np.ndarray
    mean_joint_speed: float
    holding_actions: np.ndarray | None = None


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
    errors, speeds, actions, truncated = [], [], [], False
    while not truncated:
        if not env.viewer_open:
            return None
        action = np.asarray(policy(obs), dtype=float)
        obs, _, _, truncated, info = env.step(action)
        errors.append(info["distance_m"])
        speeds.append(float(np.mean(np.abs(obs[env.layout.qvel]))))
        actions.append(action)
    return EpisodeStats(np.array(errors[-window:]), float(np.mean(speeds[-window:])), np.array(actions[-window:]))


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


def make_eval_env(
    preset: ExperimentPreset,
    randomize: bool,
    render_mode: RenderMode | None = None,
    options: EnvOptions | None = None,
) -> SO101ReachEnv:
    """Unwrapped reach env built as the preset's training env was, unless `options` override it."""
    return make_reach_env(preset, randomize, render_mode, options)


def load_trained_policy(repo: CheckpointRepository, seed: int, algo: AlgoName) -> TrainedPolicy:
    """Load a checkpoint and its frozen normalization statistics."""
    venv = DummyVecEnv([make_env_factory(False, repo.load_preset(seed))])
    normalizer = VecNormalize.load(str(repo.normalizer_path(seed)), venv)
    normalizer.training = False
    model = ALGORITHMS[algo].load(repo.model_path(seed))
    return TrainedPolicy(model, normalizer, venv)


@dataclass(frozen=True)
class EvalOverrides:
    """CLI evaluation overrides layered on each checkpoint's own env options.

    `perturb` adds encoder rounding and the largest delay in `RobustnessConfig`.
    """

    calibration_path: Path | None = None
    perturb: bool = False

    def apply(self, base: EnvOptions, robustness: RobustnessConfig = RobustnessConfig()) -> EnvOptions:
        """Base options with the requested calibration and perturbation."""
        options = base
        if self.calibration_path is not None:
            options = replace(options, calibration_path=str(self.calibration_path))
        if self.perturb:
            perturbation = PerturbationConfig(
                delay_steps=max(robustness.delay_steps), quantization_rad=robustness.quantization_rad
            )
            options = replace(options, perturbation=perturbation)
        return options


def evaluate_seed(
    repo: CheckpointRepository,
    seed: int,
    algo: AlgoName,
    num_episodes: int,
    cfg: TrainConfig,
    randomize: bool,
    render: bool,
    overrides: EvalOverrides = EvalOverrides(),
) -> list[EpisodeStats]:
    """Evaluate one seed in an env rebuilt from that seed's recorded preset."""
    render_mode: RenderMode | None = "human" if render and not randomize else None
    preset = repo.load_preset(seed)
    env = make_eval_env(preset, randomize, render_mode, overrides.apply(preset.env))
    policy = load_trained_policy(repo, seed, algo)
    try:
        return evaluate(env, policy, num_episodes, cfg)
    finally:
        env.close()
        policy.close()


def evaluate_trained(
    seeds: tuple[int, ...],
    algo: AlgoName,
    model_dir: Path,
    num_episodes: int,
    cfg: TrainConfig,
    randomize: bool,
    render: bool,
    overrides: EvalOverrides = EvalOverrides(),
) -> PolicyReport:
    """Evaluate every seed's checkpoint on nominal or randomized dynamics."""
    repo = CheckpointRepository(model_dir, cfg)
    per_seed = {str(s): evaluate_seed(repo, s, algo, num_episodes, cfg, randomize, render, overrides) for s in seeds}
    return summarize(per_seed)


def evaluate_baselines(
    num_episodes: int, cfg: TrainConfig, overrides: EvalOverrides = EvalOverrides()
) -> dict[str, PolicyReport]:
    """Random and IK-hold baselines on the same targets as the trained policy."""
    sim_cfg = SimConfig()
    env = make_reach_env(PRESETS["legacy"], randomize=False, options=overrides.apply(EnvOptions()))
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
    overrides: EvalOverrides = EvalOverrides(),
) -> dict:
    """Evaluate trained policies and baselines and assemble the JSON report."""
    nominal = evaluate_trained(seeds, algo, model_dir, num_episodes, cfg, False, render, overrides)
    randomized = evaluate_trained(seeds, algo, model_dir, num_episodes, cfg, True, False, overrides)
    baselines = evaluate_baselines(num_episodes, cfg, overrides)
    return {
        "overrides": {"calibration_path": str(overrides.calibration_path), "perturb": overrides.perturb},
        "trained_nominal": asdict(nominal),
        "trained_randomized_dynamics": asdict(randomized),
        "baselines": {name: asdict(report) for name, report in baselines.items()},
        "holding_window_steps": cfg.holding_window_steps,
        "episodes_per_seed": num_episodes,
        "passed": passes(nominal, baselines["ik_hold"], cfg),
    }

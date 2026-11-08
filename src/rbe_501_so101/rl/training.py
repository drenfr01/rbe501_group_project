"""Train one SAC or PPO seed on SO101Reach-v0."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from rbe_501_so101.config import PRESETS, ExperimentPreset, SacTuning, TrainConfig
from rbe_501_so101.envs import REACH_ENV_ID
from rbe_501_so101.rl.checkpoint import PeriodicCheckpoint, TrainingCheckpoint
from rbe_501_so101.rl.checkpoints import CheckpointRepository

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {"sac": SAC, "ppo": PPO}
AlgoName = Literal["sac", "ppo"]


def resolve_device(prefer_gpu: bool = True) -> str:
    """Return `cuda` when a GPU is visible; otherwise `cpu` or raise."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}")
        return "cuda"
    if prefer_gpu:
        raise RuntimeError(
            "CUDA is not available. Rebuild the Dev Container with GPU passthrough "
            "(`--gpus all`) or pass --cpu to train on the host CPU."
        )
    print("Using CPU")
    return "cpu"


def make_env_factory(randomize: bool, preset: ExperimentPreset = PRESETS["legacy"]) -> Callable[[], gym.Env]:
    """Return a thunk creating a reach env configured by `preset`."""

    def factory() -> gym.Env:
        return gym.make(REACH_ENV_ID, randomize_dynamics=randomize, task_cfg=preset.task, reward_cfg=preset.reward)

    return factory


def make_vec_env(
    preset: ExperimentPreset, *, randomize: bool, monitor_path: Path | None, gamma: float
) -> VecNormalize:
    """`n_envs` reach envs behind an episode monitor and observation normalization."""
    venv = DummyVecEnv([make_env_factory(randomize, preset) for _ in range(preset.sac.n_envs)])
    monitored = VecMonitor(venv, filename=str(monitor_path) if monitor_path else None)
    return VecNormalize(monitored, norm_obs=True, norm_reward=preset.sac.norm_reward, gamma=gamma)


def build_model(
    algo: AlgoName,
    env: VecNormalize,
    seed: int,
    cfg: TrainConfig,
    tuning: SacTuning = SacTuning(),
    device: str = "cuda",
) -> BaseAlgorithm:
    """Construct SAC (default) or PPO with the configured hyperparameters."""
    kwargs: dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": cfg.learning_rate,
        "gamma": cfg.gamma,
        "seed": seed,
        "verbose": 1,
        "device": device,
    }
    if algo == "sac":
        kwargs.update(_sac_kwargs(cfg, tuning))
    return ALGORITHMS[algo](**kwargs)


def _sac_kwargs(cfg: TrainConfig, tuning: SacTuning) -> dict[str, Any]:
    """SAC-only constructor arguments."""
    return {
        "buffer_size": tuning.buffer_size,
        "batch_size": cfg.batch_size,
        "gradient_steps": tuning.gradient_steps,
        "target_entropy": tuning.target_entropy,
        "use_sde": tuning.use_sde,
    }


def train_seed(
    seed: int,
    *,
    algo: AlgoName,
    timesteps: int,
    randomize_dynamics: bool,
    repo: CheckpointRepository,
    preset_name: str,
    cfg: TrainConfig,
    device: str = "cuda",
) -> Path:
    """Train one seed into a fresh folder and save model, normalizer, and preset."""
    repo.ensure_untrained(seed)
    preset = PRESETS[preset_name]
    repo.save_preset(seed, preset_name, preset)
    env = make_vec_env(preset, randomize=randomize_dynamics, monitor_path=repo.monitor_stem(seed), gamma=cfg.gamma)
    env.seed(seed)
    model = build_model(algo, env, seed, cfg, preset.sac, device=device)
    model.learn(total_timesteps=timesteps)
    model.save(repo.model_path(seed))
    env.save(str(repo.normalizer_path(seed)))
    env.close()
    return repo.seed_dir(seed)


def train_with_factory(
    seed: int,
    factory: Callable[[], gym.Env],
    *,
    algo: AlgoName,
    timesteps: int,
    seed_dir: Path,
    cfg: TrainConfig,
    device: str = "cuda",
) -> Path:
    """Train or resume one seed up to `timesteps` total steps."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = TrainingCheckpoint.in_dir(seed_dir, cfg)
    model, env = _resume_or_build(checkpoint, factory, algo, seed, cfg, device)
    remaining = timesteps - model.num_timesteps
    if remaining > 0:
        _learn(model, env, checkpoint, remaining, cfg)
        model.save(seed_dir / cfg.model_filename)
        env.save(str(seed_dir / cfg.vecnormalize_filename))
    else:
        print(f"{seed_dir} already has {model.num_timesteps} of {timesteps} timesteps; nothing to train")
    env.close()
    return seed_dir


def _resume_or_build(
    checkpoint: TrainingCheckpoint,
    factory: Callable[[], gym.Env],
    algo: AlgoName,
    seed: int,
    cfg: TrainConfig,
    device: str,
) -> tuple[BaseAlgorithm, VecNormalize]:
    """Load the checkpointed model and normalizer, or build fresh ones."""
    venv = DummyVecEnv([factory])
    venv.seed(seed)
    if checkpoint.exists():
        model, env = checkpoint.load(ALGORITHMS[algo], venv, device)
        print(f"Resuming from {checkpoint.root} at {model.num_timesteps} timesteps")
        return model, env
    env = VecNormalize(venv, norm_obs=True, norm_reward=False, gamma=cfg.gamma)
    return build_model(algo, env, seed, cfg, device=device), env


def _learn(model: BaseAlgorithm, env: VecNormalize, checkpoint: TrainingCheckpoint, steps: int, cfg: TrainConfig) -> None:
    """Train for `steps` and save a final checkpoint."""
    callback = PeriodicCheckpoint(checkpoint, env, cfg.checkpoint_freq_steps)
    try:
        model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callback)
    finally:
        callback.save_if_new()

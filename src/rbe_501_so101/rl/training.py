"""Train one SAC or PPO seed on SO101Reach-v0."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rbe_501_so101.config import TrainConfig
from rbe_501_so101.envs import REACH_ENV_ID

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


def make_env_factory(randomize: bool, monitor_path: Path | None = None) -> Callable[[], gym.Env]:
    """Return a thunk creating a (monitored) reach env for vectorization."""

    def factory() -> gym.Env:
        env = gym.make(REACH_ENV_ID, randomize_dynamics=randomize)
        return Monitor(env, filename=str(monitor_path) if monitor_path else None)

    return factory


def build_model(
    algo: AlgoName, env: VecNormalize, seed: int, cfg: TrainConfig, device: str = "cuda"
) -> BaseAlgorithm:
    """Construct SAC (default) or PPO with the configured hyperparameters."""
    kwargs: dict = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": cfg.learning_rate,
        "gamma": cfg.gamma,
        "seed": seed,
        "verbose": 1,
        "device": device,
    }
    if algo == "sac":
        kwargs.update(buffer_size=cfg.buffer_size, batch_size=cfg.batch_size)
    return ALGORITHMS[algo](**kwargs)


def train_seed(
    seed: int,
    *,
    algo: AlgoName,
    timesteps: int,
    randomize_dynamics: bool,
    model_dir: Path,
    cfg: TrainConfig,
    device: str = "cuda",
) -> Path:
    """Train one seed and save the model plus normalization statistics."""
    seed_dir = model_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    venv = DummyVecEnv([make_env_factory(randomize_dynamics, seed_dir / "train")])
    venv.seed(seed)
    env = VecNormalize(venv, norm_obs=True, norm_reward=False, gamma=cfg.gamma)
    model = build_model(algo, env, seed, cfg, device=device)
    model.learn(total_timesteps=timesteps)
    model.save(seed_dir / cfg.model_filename)
    env.save(seed_dir / cfg.vecnormalize_filename)
    env.close()
    return seed_dir

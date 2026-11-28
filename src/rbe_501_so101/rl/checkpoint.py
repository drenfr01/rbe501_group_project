"""Save and restore training state."""

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rbe_501_so101.config import TrainConfig

STAGING_SUFFIX = ".tmp"
STALE_SUFFIX = ".old"


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Model, replay buffer, and normalization state for one seed."""

    root: Path
    log: Path
    cfg: TrainConfig

    @classmethod
    def in_dir(cls, seed_dir: Path, cfg: TrainConfig) -> Self:
        """Checkpoint location of one seed folder."""
        return cls(root=seed_dir / cfg.checkpoint_dirname, log=seed_dir / cfg.checkpoint_log_filename, cfg=cfg)

    @property
    def model(self) -> Path:
        """Saved policy and optimizer state."""
        return self.root / self.cfg.model_filename

    @property
    def vecnormalize(self) -> Path:
        """Saved observation normalization statistics."""
        return self.root / self.cfg.vecnormalize_filename

    @property
    def replay_buffer(self) -> Path:
        """Saved replay buffer (off-policy algorithms only)."""
        return self.root / self.cfg.replay_buffer_filename

    def exists(self) -> bool:
        """True when a complete checkpoint is available to resume from."""
        self._recover_interrupted_swap()
        return self.model.exists() and self.vecnormalize.exists()

    def save(self, model: BaseAlgorithm, env: VecNormalize) -> None:
        """Write a complete checkpoint of `model` and `env`, then record its timestep."""
        staging = self._sibling(STAGING_SUFFIX)
        shutil.rmtree(staging, ignore_errors=True)
        replace(self, root=staging)._write(model, env)
        self._swap_in(staging)
        self._append_log(model.num_timesteps)

    def load(self, algo: type[BaseAlgorithm], venv: DummyVecEnv, device: str) -> tuple[BaseAlgorithm, VecNormalize]:
        """Restore the model, replay buffer, and normalization state."""
        env = VecNormalize.load(str(self.vecnormalize), venv)
        env.training = True
        model = algo.load(self.model, env=env, device=device)
        if isinstance(model, OffPolicyAlgorithm) and self.replay_buffer.exists():
            model.load_replay_buffer(self.replay_buffer)
        return model, env

    def _write(self, model: BaseAlgorithm, env: VecNormalize) -> None:
        """Save every file of the checkpoint into `root`."""
        self.root.mkdir(parents=True)
        model.save(self.model)
        env.save(str(self.vecnormalize))
        if isinstance(model, OffPolicyAlgorithm):
            model.save_replay_buffer(self.replay_buffer)

    def _swap_in(self, staging: Path) -> None:
        """Replace `root` with the completed `staging` folder."""
        stale = self._sibling(STALE_SUFFIX)
        shutil.rmtree(stale, ignore_errors=True)
        if self.root.exists():
            self.root.rename(stale)
        staging.rename(self.root)
        shutil.rmtree(stale, ignore_errors=True)

    def _recover_interrupted_swap(self) -> None:
        """Restore the previous checkpoint after an interrupted swap."""
        stale = self._sibling(STALE_SUFFIX)
        if not self.root.exists() and stale.exists():
            stale.rename(self.root)

    def _append_log(self, num_timesteps: int) -> None:
        """Add one timestep to the save log."""
        saves = json.loads(self.log.read_text()) if self.log.exists() else []
        self.log.write_text(json.dumps([*saves, num_timesteps]))

    def _sibling(self, suffix: str) -> Path:
        """Path next to `root` with `suffix` appended to its name."""
        return self.root.with_name(self.root.name + suffix)


class PeriodicCheckpoint(BaseCallback):
    """Save training state every `freq_steps` steps."""

    def __init__(self, checkpoint: TrainingCheckpoint, env: VecNormalize, freq_steps: int) -> None:
        super().__init__()
        self._checkpoint = checkpoint
        self._env = env
        self._freq_steps = freq_steps
        self._last_saved = 0

    def _on_training_start(self) -> None:
        """Count the interval from where this (possibly resumed) run starts."""
        self._last_saved = self.model.num_timesteps

    def _on_step(self) -> bool:
        """Save when a full interval has passed; never stop training."""
        if self.num_timesteps - self._last_saved >= self._freq_steps:
            self.save()
        return True

    def save(self) -> None:
        """Save a checkpoint at the current timestep."""
        self._checkpoint.save(self.model, self._env)
        self._last_saved = self.model.num_timesteps

    def save_if_new(self) -> None:
        """Save unless the latest checkpoint is already at the current timestep."""
        if self.model.num_timesteps != self._last_saved:
            self.save()

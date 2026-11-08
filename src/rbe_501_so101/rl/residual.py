"""Residual RL on top of the scripted pick-and-lift controller."""

from typing import Any

import gymnasium as gym
import numpy as np

from rbe_501_so101.config import ResidualConfig
from rbe_501_so101.control import ScriptedPickLift


class ResidualPickLift(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Apply `clip(scripted_action + scale * policy_action, -1, 1)`."""

    def __init__(self, env: gym.Env, controller: ScriptedPickLift, cfg: ResidualConfig = ResidualConfig()) -> None:
        super().__init__(env)
        self._controller = controller
        self._scale = cfg.scale
        self._obs: np.ndarray | None = None
        self._base: np.ndarray | None = None
        self.last_executed_action = np.zeros(env.action_space.shape, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the env and restart the scripted controller."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._controller.reset()
        self._obs, self._base = obs, None
        return obs, info

    def base_action(self) -> np.ndarray:
        """Scripted action for the current observation (computed once per step)."""
        if self._obs is None:
            raise RuntimeError("Call reset() before stepping")
        if self._base is None:
            self._base = self._controller(self._obs)
        return self._base

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Add the scaled policy action to the scripted one and step the env."""
        residual = self._scale * np.asarray(action, dtype=np.float32)
        executed = np.clip(self.base_action() + residual, -1.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(executed)
        self._obs, self._base = obs, None
        self.last_executed_action = executed
        return obs, reward, terminated, truncated, info

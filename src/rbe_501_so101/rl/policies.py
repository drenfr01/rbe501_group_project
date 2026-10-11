"""Evaluation policies: trained checkpoint, random actions, and IK-hold."""

from typing import Protocol

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from rbe_501_so101.config import ReachTaskConfig
from rbe_501_so101.envs.types import ObservationLayout, RobotState
from rbe_501_so101.kinematics import PositionIKSolver


class EvalPolicy(Protocol):
    """A policy evaluated on raw (unnormalized) observations."""

    def reset(self, target: np.ndarray, state: RobotState) -> None:
        """Prepare for a new episode after the environment has been reset."""
        ...

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return an action in [-1, 1]."""
        ...


class TrainedPolicy:
    """SB3 model behind saved VecNormalize observation statistics."""

    def __init__(self, model: BaseAlgorithm, normalizer: VecNormalize, venv: VecEnv) -> None:
        self._model = model
        self._normalizer = normalizer
        self._venv = venv

    def reset(self, target: np.ndarray, state: RobotState) -> None:
        """Stateless policy; nothing to reset."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action for the normalized observation."""
        action, _ = self._model.predict(self._normalizer.normalize_obs(obs), deterministic=True)
        return action

    def close(self) -> None:
        """Release the dummy env used only to load normalization statistics."""
        self._venv.close()


class RandomPolicy:
    """Uniformly random actions."""

    def __init__(self, seed: int, n_arm: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._n_arm = n_arm

    def reset(self, target: np.ndarray, state: RobotState) -> None:
        """Stateless policy; nothing to reset."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Random action in [-1, 1]."""
        return self._rng.uniform(-1.0, 1.0, size=self._n_arm).astype(np.float32)


class IKHoldPolicy:
    """Solve IK for the target, drive the command to that pose, never correct sag."""

    def __init__(
        self,
        solver: PositionIKSolver,
        task: ReachTaskConfig,
        layout: ObservationLayout,
    ) -> None:
        self._solver = solver
        self._delta_max = task.delta_q_max_rad
        self._layout = layout
        self._q_goal = np.zeros(layout.n_arm)

    def reset(self, target: np.ndarray, state: RobotState) -> None:
        """Solve IK from the current commanded pose for the new target."""
        self._q_goal = self._solver.solve(target, state.q_cmd).q[: self._layout.n_arm]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Rate-limited step of the command toward the IK pose."""
        q_cmd = obs[self._layout.qpos] + obs[self._layout.cmd_err]
        return np.clip((self._q_goal - q_cmd) / self._delta_max, -1.0, 1.0).astype(np.float32)

"""Sanity reaching task: drive the SO-101 gripper site to a reachable 3D target."""

from dataclasses import asdict, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rbe_501_so101.config import (
    DynamicsRandomizationConfig,
    ReachTaskConfig,
    RenderConfig,
    RewardConfig,
    SimConfig,
    TargetSamplerConfig,
)
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend, RenderMode
from rbe_501_so101.envs.protocols import (
    Renderable,
    RewardFunction,
    RobotBackend,
    TargetMarker,
    TargetSampler,
)
from rbe_501_so101.envs.rewards import ReachReward
from rbe_501_so101.envs.targets import ReachableTargetSampler
from rbe_501_so101.envs.types import RewardTerms, RobotState

CARTESIAN_DIM = 3
OBS_JOINT_BLOCKS = 4
OBS_CARTESIAN_BLOCKS = 2
# Gymnasium's env checker warns on infinite Box bounds; the float32 maximum is
# effectively unbounded while keeping check_env warning-free.
OBS_BOUND = float(np.finfo(np.float32).max)


class SO101ReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """Reach a sampled target with the 5 arm joints while the gripper is held fixed.

    Actions in [-1, 1] are scaled to relative joint-command changes. The
    observation is [q, q_dot, q_cmd - q, a_prev, p_tcp, p_target - p_tcp].
    Episodes never terminate early; they truncate after `max_episode_steps`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RenderConfig().render_fps}

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        backend: RobotBackend | None = None,
        sampler: TargetSampler | None = None,
        reward_fn: RewardFunction | None = None,
        sim_cfg: SimConfig = SimConfig(),
        task_cfg: ReachTaskConfig = ReachTaskConfig(),
        sampler_cfg: TargetSamplerConfig = TargetSamplerConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
        randomization_cfg: DynamicsRandomizationConfig = DynamicsRandomizationConfig(),
        randomize_dynamics: bool = False,
    ) -> None:
        self.render_mode = render_mode
        self._task = task_cfg
        self._n_arm = len(sim_cfg.arm_joints)
        if randomize_dynamics:
            randomization_cfg = replace(randomization_cfg, enabled=True)
        self.backend = backend or self._default_backend(sim_cfg, randomization_cfg)
        self._sampler = sampler or ReachableTargetSampler.from_config(sim_cfg, sampler_cfg)
        self._reward_fn = reward_fn or ReachReward(reward_cfg)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._n_arm,), dtype=np.float32)
        obs_dim = OBS_JOINT_BLOCKS * self._n_arm + OBS_CARTESIAN_BLOCKS * CARTESIAN_DIM
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(obs_dim,), dtype=np.float32)
        self._prev_action = np.zeros(self._n_arm, dtype=np.float32)
        self._step_count = 0
        self._state: RobotState | None = None
        self._target = np.zeros(CARTESIAN_DIM)

    @property
    def state(self) -> RobotState:
        """Most recent robot state reported by the backend."""
        if self._state is None:
            raise RuntimeError("Call reset() before accessing the state")
        return self._state

    @property
    def target(self) -> np.ndarray:
        """Current world-frame target position (m)."""
        return self._target

    @property
    def viewer_open(self) -> bool:
        """False once a human viewer window has been closed."""
        return not isinstance(self.backend, Renderable) or self.backend.is_open()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sample a new target, return the robot to home, and zero the previous action."""
        super().reset(seed=seed)
        self._target = self._sampler.sample(self.np_random)
        self._state = self.backend.reset(self.np_random)
        if isinstance(self.backend, TargetMarker):
            self.backend.set_target_marker(self._target)
        self._prev_action = np.zeros(self._n_arm, dtype=np.float32)
        self._step_count = 0
        self._render_human()
        return self._observation(), {"distance_m": self._distance(), "target": self._target.copy()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a scaled delta command, advance one control period, and score it."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.backend.apply_delta(action.astype(float) * self._task.delta_q_max_rad)
        self._state = self.backend.step()
        terms = self._reward_fn(self._state, self._target, action)
        self._prev_action = action
        self._step_count += 1
        truncated = self._step_count >= self._task.max_episode_steps
        self._render_human()
        return self._observation(), terms.total, False, truncated, self._info(terms)

    def render(self) -> np.ndarray | None:
        """Delegate rendering to the backend if it supports it."""
        if isinstance(self.backend, Renderable):
            return self.backend.render()
        return None

    def close(self) -> None:
        """Release backend resources."""
        self.backend.close()

    def _default_backend(
        self, sim_cfg: SimConfig, randomization_cfg: DynamicsRandomizationConfig
    ) -> MujocoSo101Backend:
        """MuJoCo backend whose home gripper angle is the task's hold angle."""
        home = (*sim_cfg.initial_deg[: self._n_arm], self._task.gripper_hold_deg)
        return MujocoSo101Backend(replace(sim_cfg, initial_deg=home), randomization_cfg, self.render_mode)

    def _observation(self) -> np.ndarray:
        """Build the 26-dim float32 observation from the latest state."""
        s, n = self.state, self._n_arm
        parts = (s.qpos[:n], s.qvel[:n], (s.q_cmd - s.qpos)[:n], self._prev_action, s.tcp_pos, self._target - s.tcp_pos)
        return np.concatenate(parts).astype(np.float32)

    def _distance(self) -> float:
        """Euclidean distance (m) from the gripper site to the target."""
        return float(np.linalg.norm(self.state.tcp_pos - self._target))

    def _info(self, terms: RewardTerms) -> dict[str, Any]:
        """Step info: distance plus each reward term, prefixed with `reward_`."""
        info: dict[str, Any] = {f"reward_{k}": v for k, v in asdict(terms).items()}
        info["distance_m"] = terms.distance
        return info

    def _render_human(self) -> None:
        """Push a frame to the viewer when running in human mode."""
        if self.render_mode == "human":
            self.render()

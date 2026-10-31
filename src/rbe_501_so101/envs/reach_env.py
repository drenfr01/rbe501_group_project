"""Sanity reaching task: drive the SO-101 gripper site to a reachable 3D target."""

from dataclasses import asdict, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rbe_501_so101.config import (
    DynamicsRandomizationConfig,
    IKConfig,
    ReachTaskConfig,
    RenderConfig,
    RewardConfig,
    SimConfig,
    TargetSamplerConfig,
)
from rbe_501_so101.envs.actions import make_action_mapper
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend, RenderMode
from rbe_501_so101.envs.protocols import (
    JointGoalSolver,
    Renderable,
    RewardFunction,
    RobotBackend,
    TargetMarker,
    TargetSampler,
)
from rbe_501_so101.envs.rewards import ReachReward
from rbe_501_so101.envs.targets import ReachableTargetSampler
from rbe_501_so101.envs.types import ObservationLayout, RewardTerms, RobotState
from rbe_501_so101.kinematics import PositionIKSolver

# Gymnasium's env checker warns on infinite Box bounds; the float32 maximum is
# effectively unbounded while keeping check_env warning-free.
OBS_BOUND = float(np.finfo(np.float32).max)


def task_home_deg(sim_cfg: SimConfig, task_cfg: ReachTaskConfig) -> tuple[float, ...]:
    """Home pose with the gripper at the task's hold angle (degrees)."""
    n_arm = len(sim_cfg.arm_joints)
    return (*sim_cfg.initial_deg[:n_arm], task_cfg.gripper_hold_deg)


class SO101ReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """Reach a sampled target with the 5 arm joints while the gripper is held fixed.

    Actions in [-1, 1] become joint-command changes through the configured
    action mapper (relative deltas, or a residual on an IK goal solved at
    reset). The observation is packed by `ObservationLayout`. Episodes never
    terminate early; they truncate after `max_episode_steps`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RenderConfig().render_fps}

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        backend: RobotBackend | None = None,
        sampler: TargetSampler | None = None,
        reward_fn: RewardFunction | None = None,
        sim_cfg: SimConfig | None = None,
        task_cfg: ReachTaskConfig | None = None,
        sampler_cfg: TargetSamplerConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        randomization_cfg: DynamicsRandomizationConfig | None = None,
        randomize_dynamics: bool = False,
        goal_solver: JointGoalSolver | None = None,
    ) -> None:
        self._reject_conflicting_backend(backend, sim_cfg, randomization_cfg, randomize_dynamics)
        sim_cfg = sim_cfg or SimConfig()
        task_cfg = task_cfg or ReachTaskConfig()
        sampler_cfg = sampler_cfg or TargetSamplerConfig()
        reward_cfg = reward_cfg or RewardConfig()
        randomization_cfg = randomization_cfg or DynamicsRandomizationConfig()
        if randomize_dynamics:
            randomization_cfg = replace(randomization_cfg, enabled=True)
        self.render_mode = render_mode
        self._sim = sim_cfg
        self._task = task_cfg
        self._layout = self._build_layout(sim_cfg, task_cfg)
        self.backend = backend or self._default_backend(sim_cfg, randomization_cfg)
        self._sampler = sampler or self._default_sampler(sim_cfg, sampler_cfg, task_cfg)
        self._reward_fn = reward_fn or ReachReward(reward_cfg)
        self._action_mapper = make_action_mapper(task_cfg)
        self.goal_solver = goal_solver or self._default_goal_solver(sim_cfg)
        self._init_spaces()
        self._reset_episode_state()

    @staticmethod
    def _build_layout(sim_cfg: SimConfig, task_cfg: ReachTaskConfig) -> ObservationLayout:
        """Observation layout with the extras the task configuration asks for."""
        return ObservationLayout(
            n_arm=len(sim_cfg.arm_joints),
            include_fine_error=task_cfg.fine_error_obs,
            include_joint_goal=task_cfg.action_mode == "residual_ik",
        )

    @staticmethod
    def _reject_conflicting_backend(
        backend: RobotBackend | None,
        sim_cfg: SimConfig | None,
        randomization_cfg: DynamicsRandomizationConfig | None,
        randomize_dynamics: bool,
    ) -> None:
        """Fail if builder kwargs would have constructed a different backend."""
        if backend is None:
            return
        if sim_cfg is not None or randomization_cfg is not None or randomize_dynamics:
            raise ValueError(
                "Do not pass sim_cfg, randomization_cfg, or randomize_dynamics "
                "when injecting a backend"
            )

    def _init_spaces(self) -> None:
        """Action and observation boxes from the observation layout."""
        n_arm, obs_dim = self._layout.n_arm, self._layout.size
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_arm,), dtype=np.float32)
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(obs_dim,), dtype=np.float32)

    def _reset_episode_state(self) -> None:
        """Zero per-episode fields that `reset()` also restores."""
        self._prev_action = np.zeros(self._layout.n_arm, dtype=np.float32)
        self._step_count = 0
        self._state: RobotState | None = None
        self._target = np.zeros(self._layout.cartesian_dim)
        self._joint_goal: np.ndarray | None = None

    @property
    def layout(self) -> ObservationLayout:
        """Named slices for the packed observation vector."""
        return self._layout

    @property
    def control_dt(self) -> float:
        """Duration of one `step()` in seconds."""
        return self._sim.control_dt

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
        self._joint_goal = self._solve_joint_goal()
        self._action_mapper.reset(self._joint_goal)
        self._prev_action = np.zeros(self._layout.n_arm, dtype=np.float32)
        self._step_count = 0
        self._render_human()
        return self._observation(), {"distance_m": self._distance(), "target": self._target.copy()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Map the action to a command change, advance one control period, and score it."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.backend.apply_delta(self._action_mapper.to_delta(action, self.state))
        self._state = self.backend.step()
        terms = self._reward_fn(self._state, self._target, action, self._prev_action)
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
        home = task_home_deg(sim_cfg, self._task)
        return MujocoSo101Backend(replace(sim_cfg, initial_deg=home), randomization_cfg, self.render_mode)

    def _default_sampler(
        self, sim_cfg: SimConfig, sampler_cfg: TargetSamplerConfig, task_cfg: ReachTaskConfig
    ) -> ReachableTargetSampler:
        """Sampler that reuses the backend model when the backend is MuJoCo."""
        model = getattr(self.backend, "model", None)
        ids = getattr(self.backend, "ids", None)
        if model is not None and ids is not None:
            return ReachableTargetSampler(
                model, ids, sim_cfg, sampler_cfg, gripper_hold_deg=task_cfg.gripper_hold_deg
            )
        return ReachableTargetSampler.from_config(
            sim_cfg, sampler_cfg, gripper_hold_deg=task_cfg.gripper_hold_deg
        )

    def _default_goal_solver(self, sim_cfg: SimConfig) -> JointGoalSolver | None:
        """IK solver for residual mode, sharing the backend model when possible."""
        if self._task.action_mode != "residual_ik":
            return None
        model = getattr(self.backend, "model", None)
        ids = getattr(self.backend, "ids", None)
        if model is not None and ids is not None:
            return PositionIKSolver(model, ids, sim_cfg, IKConfig())
        return PositionIKSolver.from_config(sim_cfg, IKConfig())

    def _solve_joint_goal(self) -> np.ndarray | None:
        """IK joint goal for the current target from the commanded pose, if used."""
        if self.goal_solver is None:
            return None
        return np.asarray(self.goal_solver.joint_goal(self._target, self.state.q_cmd), dtype=float)

    def _observation(self) -> np.ndarray:
        """Build the float32 observation from the latest state."""
        s, n = self.state, self._layout.n_arm
        rel_target = self._target - s.tcp_pos
        return self._layout.pack(
            s.qpos,
            s.qvel,
            (s.q_cmd - s.qpos)[:n],
            self._prev_action,
            s.tcp_pos,
            rel_target,
            fine_error=self._fine_error(rel_target),
            joint_goal=self._joint_goal,
        )

    def _fine_error(self, rel_target: np.ndarray) -> np.ndarray:
        """Target error clipped to `fine_error_clip_m` and scaled to [-1, 1]."""
        clip = self._task.fine_error_clip_m
        return np.clip(rel_target, -clip, clip) / clip

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

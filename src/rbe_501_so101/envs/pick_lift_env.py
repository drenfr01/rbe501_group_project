"""Pick-and-lift task: grasp the cube and raise it, with the object mass as a training parameter."""

from dataclasses import asdict, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rbe_501_so101.config import ObjectMassConfig, PickLiftConfig, RenderConfig
from rbe_501_so101.envs.mujoco_backend import RenderMode
from rbe_501_so101.envs.pick_lift_backend import MujocoPickLiftBackend
from rbe_501_so101.envs.placement import TablePlacer
from rbe_501_so101.envs.protocols import GraspBackend, ObjectPlacer, PickLiftRewardFunction, Renderable
from rbe_501_so101.envs.rewards import PickLiftReward
from rbe_501_so101.envs.types import GraspState, PickLiftObservationLayout, PickLiftRewardTerms

OBS_BOUND = float(np.finfo(np.float32).max)
class SO101PickLiftEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium task for grasping and lifting a cube."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RenderConfig().render_fps}

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        mass_cfg: ObjectMassConfig | None = None,
        cfg: PickLiftConfig | None = None,
        backend: GraspBackend | None = None,
        placer: ObjectPlacer | None = None,
        reward_fn: PickLiftRewardFunction | None = None,
    ) -> None:
        cfg = cfg or PickLiftConfig()
        self._cfg = replace(cfg, mass=mass_cfg) if mass_cfg is not None else cfg
        self.render_mode = render_mode
        self._n_arm = len(self._cfg.sim.arm_joints)
        self._layout = PickLiftObservationLayout(len(self._cfg.sim.joints))
        self.backend = backend or self._default_backend()
        self._placer = placer or self._default_placer()
        self._reward_fn = reward_fn or PickLiftReward(self._cfg.reward, self._cfg.task.lift_success_m)
        self._init_spaces()
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._step_count = 0
        self._state: GraspState | None = None

    @property
    def layout(self) -> PickLiftObservationLayout:
        """Named blocks of the observation vector."""
        return self._layout

    @property
    def config(self) -> PickLiftConfig:
        """Settings this environment was built with."""
        return self._cfg

    @property
    def grasp_state(self) -> GraspState:
        """Most recent grasp state reported by the backend."""
        if self._state is None:
            raise RuntimeError("Call reset() before accessing the state")
        return self._state

    @property
    def control_dt(self) -> float:
        """Duration of one `step()` in seconds."""
        return self._cfg.sim.control_dt

    @property
    def viewer_open(self) -> bool:
        """False once a human viewer window has been closed."""
        return not isinstance(self.backend, Renderable) or self.backend.is_open()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Home the arm, then place a cube with a freshly sampled pose and mass."""
        super().reset(seed=seed)
        self.backend.reset(self.np_random)
        p_s, yaw = self._placer.sample(self.np_random)
        self.backend.place_object(p_s, yaw, self._cfg.mass.sample(self.np_random))
        self._state = self.backend.grasp_state()
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._step_count = 0
        self._render_human()
        return self._observation(), self._base_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply scaled arm and gripper deltas, advance one control period, and score it."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        task = self._cfg.task
        self.backend.apply_delta(action[: self._n_arm].astype(float) * task.delta_q_max_rad)
        self.backend.apply_gripper_delta(float(action[self._n_arm]) * task.delta_gripper_max_rad)
        self.backend.step()
        self._state = self.backend.grasp_state()
        terms = self._reward_fn(self._state, action)
        self._prev_action = action
        self._step_count += 1
        truncated = self._step_count >= task.max_episode_steps
        self._render_human()
        return self._observation(), terms.total, False, truncated, self._step_info(terms)

    def render(self) -> np.ndarray | None:
        """Delegate rendering to the backend if it supports it."""
        if isinstance(self.backend, Renderable):
            return self.backend.render()
        return None

    def close(self) -> None:
        """Release backend resources."""
        self.backend.close()

    def _init_spaces(self) -> None:
        """Action and observation boxes."""
        n_joints = len(self._cfg.sim.joints)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(self._layout.size,), dtype=np.float32)

    def _default_backend(self) -> MujocoPickLiftBackend:
        """MuJoCo backend built from this env's configuration."""
        cfg = self._cfg
        return MujocoPickLiftBackend(cfg.sim, cfg.gripper, cfg.object, cfg.contact, self.render_mode)

    def _default_placer(self) -> TablePlacer:
        """Placer around the backend's pan axis at the backend's resting height."""
        backend = self.backend
        if not isinstance(backend, MujocoPickLiftBackend):
            raise TypeError("Pass a placer when injecting a non-MuJoCo backend")
        return TablePlacer(self._cfg.placement, backend.pan_axis_xy, backend.rest_z)

    def _observation(self) -> np.ndarray:
        """Pack the latest grasp state into the float32 observation."""
        s = self.grasp_state
        robot = s.robot
        return np.concatenate(
            (
                robot.qpos,
                robot.qvel,
                robot.q_cmd - robot.qpos,
                self._prev_action,
                s.p_pinch,
                s.obj.p_s,
                s.obj.p_s - s.p_pinch,
            )
        ).astype(np.float32)

    def _base_info(self) -> dict[str, Any]:
        """Mass, distance, lift height, grasp, and success flags for the current state."""
        s = self.grasp_state
        return {
            "mass_kg": s.obj.mass_kg,
            "distance_m": float(np.linalg.norm(s.obj.p_s - s.p_pinch)),
            "lift_m": s.obj.lift_m,
            "grasped": s.grasped,
            "is_success": self._reward_fn.is_success(s),
        }

    def _step_info(self, terms: PickLiftRewardTerms) -> dict[str, Any]:
        """Base info plus each reward term, prefixed with `reward_`."""
        info = self._base_info()
        info.update({f"reward_{k}": v for k, v in asdict(terms).items()})
        return info

    def _render_human(self) -> None:
        """Push a frame to the viewer when running in human mode."""
        if self.render_mode == "human":
            self.render()

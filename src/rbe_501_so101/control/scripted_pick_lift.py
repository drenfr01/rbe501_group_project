"""Scripted grasp with pinch-point IK and rate-limited joint commands."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import numpy as np

from rbe_501_so101.config import GripperConfig, PickLiftTaskConfig, PinchIKConfig, ScriptedPickLiftConfig
from rbe_501_so101.envs.pick_lift_backend import MujocoPickLiftBackend
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from rbe_501_so101.envs.types import PickLiftObservationLayout
from rbe_501_so101.kinematics.pinch_ik import PinchIKSolver

UP_S = np.array([0.0, 0.0, 1.0])


class Phase(StrEnum):
    """Stages of the scripted grasp, in execution order."""

    APPROACH = "approach"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"


@dataclass(frozen=True)
class PhaseTarget:
    """Joint-space goal for one phase and the per-step rates used to reach it."""

    q_arm: np.ndarray
    gripper_rad: float
    arm_rate_rad: float
    gripper_rate_rad: float


class ScriptedPickLift:
    """Approach, descend, close, and lift using phase-specific IK targets."""

    def __init__(
        self,
        solver: PinchIKSolver,
        layout: PickLiftObservationLayout,
        task: PickLiftTaskConfig,
        gripper: GripperConfig,
        cfg: ScriptedPickLiftConfig = ScriptedPickLiftConfig(),
    ) -> None:
        self._solver = solver
        self._layout = layout
        self._task = task
        self._gripper = gripper
        self._cfg = cfg
        self.reset()

    @classmethod
    def for_env(cls, env: SO101PickLiftEnv, cfg: ScriptedPickLiftConfig = ScriptedPickLiftConfig()) -> Self:
        """Controller whose IK shares the env's MuJoCo model and pinch offset."""
        backend = env.unwrapped.backend
        if not isinstance(backend, MujocoPickLiftBackend):
            raise TypeError("ScriptedPickLift needs the MuJoCo pick-lift backend for IK")
        env_cfg = env.unwrapped.config
        p_b = np.asarray(env_cfg.gripper.pinch_point_b(env_cfg.object.side_m))
        solver = PinchIKSolver(backend.model, backend.ids, PinchIKConfig(), p_b)
        return cls(solver, env.unwrapped.layout, env_cfg.task, env_cfg.gripper, cfg)

    @property
    def phase(self) -> Phase | None:
        """Phase of the most recent action, or None before the first call."""
        return self._phase

    def reset(self) -> None:
        """Start a new episode from the first phase."""
        self._step = 0
        self._phase: Phase | None = None
        self._target: PhaseTarget | None = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Normalized action in [-1, 1]^6 that moves the command toward the phase target."""
        phase = self._phase_at(self._step)
        if phase != self._phase:
            self._target = self._enter(phase, obs)
            self._phase = phase
        self._step += 1
        return self._track(obs, self._target)

    def _enter(self, phase: Phase, obs: np.ndarray) -> PhaseTarget:
        """Solve the joint-space goal for a phase from the current observation."""
        q_cmd = self._q_cmd(obs)
        p_obj = obs[self._layout.object_pos].astype(float)
        cfg, grip = self._cfg, self._gripper
        open_rad, close_rad = np.deg2rad(grip.open_deg), np.deg2rad(grip.close_deg)
        match phase:
            case Phase.APPROACH:
                q = self._ik(p_obj + cfg.pregrasp_height_m * UP_S, q_cmd, top_down=True)
                return PhaseTarget(q, open_rad, cfg.arm_rate_rad, cfg.close_rate_rad)
            case Phase.DESCEND:
                q = self._ik(p_obj + cfg.grasp_height_m * UP_S, q_cmd, top_down=True)
                return PhaseTarget(q, open_rad, cfg.arm_rate_rad, cfg.close_rate_rad)
            case Phase.CLOSE:
                return PhaseTarget(self._target.q_arm, close_rad, cfg.arm_rate_rad, cfg.close_rate_rad)
            case Phase.LIFT:
                q = self._ik(p_obj + cfg.lift_height_m * UP_S, q_cmd, top_down=False)
                return PhaseTarget(q, close_rad, cfg.lift_rate_rad, cfg.close_rate_rad)

    def _ik(self, goal: np.ndarray, q_cmd: np.ndarray, top_down: bool) -> np.ndarray:
        """Arm joints that put the pinch point on `goal` (best effort if IK misses)."""
        return self._solver.solve(goal, q_cmd, top_down).q[: self._n_arm]

    def _track(self, obs: np.ndarray, target: PhaseTarget) -> np.ndarray:
        """Rate-limited step of the commanded joints toward the target, normalized to [-1, 1]."""
        q_cmd = self._q_cmd(obs)
        arm = self._limited(target.q_arm - q_cmd[: self._n_arm], target.arm_rate_rad, self._task.delta_q_max_rad)
        grip_error = np.array([target.gripper_rad - q_cmd[self._n_arm]])
        grip = self._limited(grip_error, target.gripper_rate_rad, self._task.delta_gripper_max_rad)
        return np.concatenate([arm, grip]).astype(np.float32)

    @staticmethod
    def _limited(error: np.ndarray, rate: float, full_scale: float) -> np.ndarray:
        """Clip a joint error to +-rate, then express it as a fraction of the env's full-scale delta."""
        return np.clip(error, -rate, rate) / full_scale

    def _q_cmd(self, obs: np.ndarray) -> np.ndarray:
        """Commanded joint angles recovered from the observation: q + (q_cmd - q)."""
        return (obs[self._layout.q] + obs[self._layout.command_error]).astype(float)

    @property
    def _n_arm(self) -> int:
        """Number of arm joints (the gripper is the last joint)."""
        return self._layout.n_joints - 1

    def _phase_at(self, step: int) -> Phase:
        """Choose the active phase from the three configured transition times."""
        cfg = self._cfg
        if step < cfg.approach_steps:
            return Phase.APPROACH
        if step < cfg.approach_steps + cfg.descend_steps:
            return Phase.DESCEND
        if step < cfg.approach_steps + cfg.descend_steps + cfg.close_steps:
            return Phase.CLOSE
        return Phase.LIFT

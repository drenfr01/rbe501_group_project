"""Evaluation policies: trained checkpoint, random actions, and classical baselines.

The classical baselines (IK hold, gravity compensation, joint integral) all
emit delta-mode actions with the same rate limit, so they must run in a
`action_mode="delta"` env.
"""

from typing import Literal, Protocol

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from rbe_501_so101.config import IKConfig, IntegralControlConfig, ReachTaskConfig, SimConfig
from rbe_501_so101.envs.calibration import ActuatorCalibration
from rbe_501_so101.envs.types import ObservationLayout, RobotState
from rbe_501_so101.kinematics import PositionIKSolver
from rbe_501_so101.kinematics.gravity import GravityTorqueModel


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
        return rate_limited_action(self._q_goal, commanded(obs, self._layout), self._delta_max)


class GravityCompPolicy:
    """Feedforward: hold `q_ik + clip(tau_g(q_ik) / kp, ±residual_max)`.

    The gravity torque at the IK pose is the torque needed to hold the pose we
    want, so no iteration is needed. `kp` defaults to the gravity model's own
    actuator stiffness; pass measured values to use a calibration.
    """

    def __init__(
        self,
        solver: PositionIKSolver,
        task: ReachTaskConfig,
        layout: ObservationLayout,
        gravity: GravityTorqueModel,
        kp: np.ndarray | None = None,
    ) -> None:
        self._solver = solver
        self._gravity = gravity
        self._kp = gravity.actuator_kp() if kp is None else np.asarray(kp, dtype=float)
        self._delta_max = task.delta_q_max_rad
        self._residual_max = task.residual_max_rad
        self._layout = layout
        self._q_goal = np.zeros(layout.n_arm)

    def reset(self, target: np.ndarray, state: RobotState) -> None:
        """Solve IK and add the predicted sag at that pose."""
        q_ik = self._solver.solve(target, state.q_cmd).q
        correction = np.clip(self._gravity.torque(q_ik) / self._kp, -self._residual_max, self._residual_max)
        self._q_goal = q_ik[: self._layout.n_arm] + correction

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Rate-limited step of the command toward the compensated pose."""
        return rate_limited_action(self._q_goal, commanded(obs, self._layout), self._delta_max)


class JointIntegralPolicy:
    """Model-free feedback: integrate each joint's remaining error into its command.

    Once the command reaches the IK pose, each step adds `gain * (q_ik - q)`
    to a correction `z` clipped to ±residual_max, and drives the command
    toward `q_ik + z`. Only measured joint angles are used.
    """

    def __init__(
        self,
        solver: PositionIKSolver,
        task: ReachTaskConfig,
        layout: ObservationLayout,
        cfg: IntegralControlConfig = IntegralControlConfig(),
    ) -> None:
        self._solver = solver
        self._cfg = cfg
        self._delta_max = task.delta_q_max_rad
        self._residual_max = task.residual_max_rad
        self._layout = layout
        self._q_ik = np.zeros(layout.n_arm)
        self._z = np.zeros(layout.n_arm)
        self._integrating = False

    @property
    def correction(self) -> np.ndarray:
        """Current integral correction (rad), a copy."""
        return self._z.copy()

    def reset(self, target: np.ndarray, state: RobotState) -> None:
        """Solve IK and zero the correction."""
        self._q_ik = self._solver.solve(target, state.q_cmd).q[: self._layout.n_arm]
        self._z = np.zeros(self._layout.n_arm)
        self._integrating = False

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Integrate (once the IK pose is commanded) and step toward the corrected pose."""
        q_cmd = commanded(obs, self._layout)
        if not self._integrating:
            self._integrating = bool(np.all(np.abs(self._q_ik - q_cmd) < self._cfg.reached_tol_rad))
        if self._integrating:
            error = self._q_ik - obs[self._layout.qpos]
            self._z = np.clip(self._z + self._cfg.gain * error, -self._residual_max, self._residual_max)
        return rate_limited_action(self._q_ik + self._z, q_cmd, self._delta_max)


BaselineName = Literal["ik_hold", "gravity_comp", "integral"]
BASELINE_NAMES: tuple[BaselineName, ...] = ("ik_hold", "gravity_comp", "integral")


def make_baseline(
    name: str,
    sim_cfg: SimConfig,
    task: ReachTaskConfig,
    layout: ObservationLayout,
    calibration: ActuatorCalibration | None = None,
    integral: IntegralControlConfig = IntegralControlConfig(),
) -> EvalPolicy:
    """Build a classical controller by name, each with its own nominal model copy."""
    solver = PositionIKSolver.from_config(sim_cfg, IKConfig())
    match name:
        case "ik_hold":
            return IKHoldPolicy(solver, task, layout)
        case "gravity_comp":
            gravity = GravityTorqueModel.from_config(sim_cfg)
            kp = None if calibration is None else calibration.kp_for(sim_cfg.arm_joints, gravity.actuator_kp())
            return GravityCompPolicy(solver, task, layout, gravity, kp)
        case "integral":
            return JointIntegralPolicy(solver, task, layout, integral)
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")


def commanded(obs: np.ndarray, layout: ObservationLayout) -> np.ndarray:
    """Commanded arm angles recovered from the observation (`q + (q_cmd - q)`)."""
    return obs[layout.qpos] + obs[layout.cmd_err]


def rate_limited_action(goal: np.ndarray, q_cmd: np.ndarray, delta_max: float) -> np.ndarray:
    """Delta-mode action that moves the command toward `goal` by at most `delta_max`."""
    return np.clip((goal - q_cmd) / delta_max, -1.0, 1.0).astype(np.float32)

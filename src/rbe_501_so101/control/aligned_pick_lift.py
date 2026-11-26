"""Opt-in scripted acquisition/lift ablations; original defaults are unchanged.

Radial yaw alignment assumes this task's TablePlacer rule (cube yaw equals its
azimuth around the pan axis). It is not a general object-orientation estimator.
Gravity feedforward changes command angles, not physical gains or gravcomp.
"""

from dataclasses import dataclass, replace
from typing import Literal, Self

import numpy as np

from rbe_501_so101.config import ScriptedPickLiftConfig
from rbe_501_so101.control.scripted_pick_lift import ScriptedPickLift, Phase, PhaseTarget
from rbe_501_so101.kinematics.gravity import GravityTorqueModel


@dataclass(frozen=True)
class MotionOptions:
    """Explicit ablation parameters; units are meters and radians."""

    name: str
    gravity_feedforward: bool = False
    radial_jaw_alignment: bool = False
    grasp_height_m: float = 0.0
    upright_lift: bool = False
    lift_reference: Literal["cube", "pinch"] = "cube"
    correction_limit_rad: float = 0.1
    yaw_rounds: int = 3
    yaw_probe_rad: float = 0.001
    minimum_yaw_derivative: float = 0.1


def quarter_turn_error(desired: float, actual: float) -> float:
    """Nearest signed square-face yaw error, in [-pi/4, pi/4)."""
    return float((desired - actual + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def compensate(q, gravity_torque, kp, limits, maximum: float) -> np.ndarray:
    """Bound q_cmd = q_ideal + gravity torque / kp to correction and joint limits."""
    correction = np.clip(gravity_torque / kp, -maximum, maximum)
    return np.clip(q + correction, limits[:, 0], limits[:, 1])


class AlignedPickLift(ScriptedPickLift):
    """Use the existing IK mathematics with optional yaw seeds and feedforward."""

    @classmethod
    def for_env(cls, env, options: MotionOptions = MotionOptions("baseline")) -> Self:
        """Construct before applying a manipulation backend adapter."""
        cfg = replace(ScriptedPickLiftConfig(), grasp_height_m=options.grasp_height_m)
        policy = super().for_env(env, cfg)
        policy.options = options
        backend = env.unwrapped.backend
        policy.gravity = GravityTorqueModel(backend.model, backend.ids)
        policy.kp = policy.gravity.actuator_kp()
        return policy

    def reset(self) -> None:
        """Clear episode diagnostics and the initial task-specific cube bearing."""
        super().reset()
        self._desired_yaw = 0.0
        self.ik_misses = 0

    def _enter(self, phase: Phase, obs: np.ndarray) -> PhaseTarget:
        """Ablate acquisition settings and lift reference/orientation separately."""
        if phase == Phase.APPROACH:
            xy = obs[self._layout.object_pos][:2] - self._solver._pan_axis_xy
            self._desired_yaw = float(np.arctan2(xy[1], xy[0]))
        changed_lift = self.options.upright_lift or self.options.lift_reference == "pinch"
        if phase != Phase.LIFT or not changed_lift:
            return super()._enter(phase, obs)
        start = obs[self._layout.object_pos].astype(float)
        if self.options.lift_reference == "pinch":
            start = self._solver.forward(obs[self._layout.q].astype(float))
        goal = start + [0, 0, self._cfg.lift_height_m]
        q = self._ik(goal, self._q_cmd(obs), self.options.upright_lift)
        return PhaseTarget(q, np.deg2rad(self._gripper.close_deg), self._cfg.lift_rate_rad, self._cfg.close_rate_rad)

    def _ik(self, goal, q_cmd, top_down) -> np.ndarray:
        """Keep baseline solving, then optionally seed jaw yaw and correct gravity sag."""
        result = self._solver.solve(goal, q_cmd, top_down)
        if top_down and self.options.radial_jaw_alignment:
            result = self._align(goal, result.q)
        self.ik_misses += int(not result.converged())
        q = result.q[: self._n_arm]
        if self.options.gravity_feedforward:
            q = compensate(q, self.gravity.torque(result.q), self.kp,
                           self._solver._arm_range, self.options.correction_limit_rad)
        return q

    def _align(self, goal, q):
        """Seed the wrist toward the nearest compatible square-face direction."""
        result = self._solver.solve(goal, q, True)
        for _ in range(self.options.yaw_rounds):
            q = result.q.copy()
            _, rotation = self._solver.tcp_pose(q)
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            probe = q.copy()
            probe[-2] += self.options.yaw_probe_rad
            _, rotated = self._solver.tcp_pose(probe)
            derivative = quarter_turn_error(float(np.arctan2(rotated[1, 0], rotated[0, 0])), yaw) / self.options.yaw_probe_rad
            if abs(derivative) < self.options.minimum_yaw_derivative:
                break
            q[-2] = np.clip(q[-2] + quarter_turn_error(self._desired_yaw, yaw) / derivative,
                             *self._solver._arm_range[-1])
            result = self._solver.solve(goal, q, True)
        return result

"""Opt-in raise-before-traverse and measured pre-close alignment experiments."""

from dataclasses import dataclass, replace
from math import radians

import numpy as np

from rbe_501_so101.control.aligned_pick_lift import AlignedPickLift, compensate
from rbe_501_so101.control.guarded_pick_lift import GuardedLift
from rbe_501_so101.control.scripted_pick_lift import Phase, PhaseTarget
from rbe_501_so101.rl.calibrated_pick_lift import outer_wrist_motion


@dataclass(frozen=True)
class ApproachConfig:
    """Meter/radian thresholds and independent switches; old timing is retained."""

    name: str = "raise_and_align"
    raise_first: bool = True
    alignment_gate: bool = True
    raise_delta_m: float = .05
    raise_above_cube_m: float = .08
    raise_tolerance_m: float = .025
    xy_tolerance_m: float = .008
    z_tolerance_m: float = .008
    tilt_tolerance_rad: float = radians(8)
    correction_enabled: bool = False
    correction_gain: float = .5
    correction_limit_m: float = .015
    correction_period: int = 3
    stable_alignment_steps: int = 3


def aligned_for_close(error: np.ndarray, tilt: float, cfg: ApproachConfig) -> bool:
    """Use measured pose, not the ideal IK residual, to authorize closing."""
    return bool(np.linalg.norm(error[:2]) <= cfg.xy_tolerance_m and
                abs(error[2]) <= cfg.z_tolerance_m and tilt <= cfg.tilt_tolerance_rad)


def corrected_bias(bias: np.ndarray, error: np.ndarray, cfg: ApproachConfig) -> np.ndarray:
    """Bound a measured Cartesian correction while leaving physical gains untouched."""
    return np.clip(bias - cfg.correction_gain * error, -cfg.correction_limit_m, cfg.correction_limit_m)


class RaisedApproach(AlignedPickLift):
    """Reuse the arm IK and final waypoints; change only acquisition staging."""

    @classmethod
    def for_env(cls, env, acquisition: ApproachConfig = ApproachConfig()):
        """Construct before the experimental servo adapter, exactly like the baseline."""
        policy = super().for_env(env, outer_wrist_motion())
        policy.acquisition = acquisition
        return policy

    def reset(self) -> None:
        """Clear stage, measurement gate and diagnostic counters between placements."""
        super().reset()
        self.stage = "initial"
        self.close_started = False
        self.aborted = False
        self.abort_reason = None
        self.raise_goal = None
        self.traverse_target = None
        self.alignment_wait_steps = 0
        self.alignment_error = None
        self.alignment_tilt = None
        self.alignment_count = 0
        self.target_bias = np.zeros(3)

    def _enter(self, phase: Phase, obs: np.ndarray) -> PhaseTarget:
        """Abort open if raising times out or alignment never permits closing."""
        if self.aborted:
            return self._target
        if phase == Phase.APPROACH:
            return self._approach(obs)
        if phase == Phase.DESCEND and self.stage == "raise":
            self.aborted, self.abort_reason = True, "raise_timeout"
            return self._target
        if phase == Phase.LIFT and self.acquisition.alignment_gate and not self.close_started:
            self.aborted, self.abort_reason = True, "alignment_timeout"
            return self._target
        target = super()._enter(phase, obs)
        if phase == Phase.CLOSE and self.acquisition.alignment_gate:
            return replace(target, gripper_rad=np.deg2rad(self._gripper.open_deg))
        return target

    def _approach(self, obs: np.ndarray) -> PhaseTarget:
        """First raise at the sensed XY with free orientation, then traverse high."""
        target = super()._enter(Phase.APPROACH, obs)
        self.stage = "traverse"
        if not self.acquisition.raise_first:
            return target
        self.traverse_target = target
        start = obs[self._layout.pinch_pos].astype(float)
        self.raise_goal = start.copy()
        self.raise_goal[2] = max(start[2] + self.acquisition.raise_delta_m,
                                 obs[self._layout.object_pos][2] + self.acquisition.raise_above_cube_m)
        self.stage = "raise"
        q = self._ik(self.raise_goal, self._q_cmd(obs), top_down=False)
        return replace(target, q_arm=q)

    def _alignment(self, obs: np.ndarray) -> bool:
        """Read delayed/quantized pose and wrist orientation available to the controller."""
        goal = obs[self._layout.object_pos].astype(float) + [0, 0, self.options.grasp_height_m]
        self.alignment_error = obs[self._layout.pinch_pos].astype(float) - goal
        _, rotation = self._solver.tcp_pose(obs[self._layout.q].astype(float))
        self.alignment_tilt = float(np.arccos(np.clip(-rotation[2, 2], -1, 1)))
        return aligned_for_close(self.alignment_error, self.alignment_tilt, self.acquisition)

    def _correct_alignment(self, obs: np.ndarray) -> None:
        """Solve a bounded target correction while the gripper remains open."""
        self.target_bias = corrected_bias(self.target_bias, self.alignment_error, self.acquisition)
        goal = obs[self._layout.object_pos].astype(float) + [0, 0, self.options.grasp_height_m]
        result = self._solver.solve(goal + self.target_bias, self._q_cmd(obs), True)
        self.ik_misses += int(not result.converged())
        q = compensate(result.q[:self._n_arm], self.gravity.torque(result.q), self.kp,
                       self._solver._arm_range, self.options.correction_limit_rad)
        self._target = replace(self._target, q_arm=q)

    def _closing_action(self, obs: np.ndarray) -> np.ndarray:
        """Require several aligned measurements; otherwise wait or correct within the window."""
        ready = self._alignment(obs)
        self.alignment_count = self.alignment_count + 1 if ready else 0
        self.close_started = self.alignment_count >= self.acquisition.stable_alignment_steps
        if self.close_started:
            self._target = replace(self._target, gripper_rad=np.deg2rad(self._gripper.close_deg))
        else:
            self.alignment_wait_steps += 1
            if not ready and self.acquisition.correction_enabled:
                if self.alignment_wait_steps % self.acquisition.correction_period == 0:
                    self._correct_alignment(obs)
        return self._track(obs, self._target)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Preserve the old phase clock; wait open for alignment within the close window."""
        if self.stage == "raise" and self._phase == Phase.APPROACH:
            if obs[self._layout.pinch_pos][2] >= self.raise_goal[2] - self.acquisition.raise_tolerance_m:
                self._target, self.stage = self.traverse_target, "traverse"
        action = super().__call__(obs)
        if not self.aborted and self._phase == Phase.CLOSE and self.acquisition.alignment_gate and not self.close_started:
            action = self._closing_action(obs)
        return action


class AcquisitionGuard(GuardedLift):
    """Do not let the existing lift guard close the jaw after an acquisition abort."""

    def __init__(self, env, controller: RaisedApproach) -> None:
        super().__init__(env)
        self.controller = controller

    def step(self, action):
        """Hold arm commands and command the jaw open for a latched acquisition abort."""
        if not self.controller.aborted:
            return super().step(action)
        raw = self.unwrapped
        command = np.zeros_like(action)
        error = np.deg2rad(raw.config.gripper.open_deg) - raw.grasp_state.robot.q_cmd[-1]
        command[-1] = np.clip(error / raw.config.task.delta_gripper_max_rad, -1, 1)
        self.gate.aborted = True
        self.steps += 1
        obs, reward, terminated, truncated, info = self.env.step(command)
        info.update(lift_released=False, grasp_abort=True, acquisition_abort=True)
        return obs, reward, terminated, truncated, info

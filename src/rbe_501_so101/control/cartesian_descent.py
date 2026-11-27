"""Experimental measured-alignment and sampled-clearance Cartesian descent."""

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from rbe_501_so101.control.aligned_pick_lift import compensate
from rbe_501_so101.control.raised_approach import AcquisitionGuard, ApproachConfig, RaisedApproach
from rbe_501_so101.control.scripted_pick_lift import Phase
from rbe_501_so101.rl.lift_residual_limits import CommandPoseChecker, ResidualLimits


@dataclass(frozen=True)
class DescentConfig:
    """Fixed scene units: meters, radians, and control-step counts."""

    name: str = "cartesian_checked"
    grasp_height_m: float = 0.
    xy_tolerance_m: float = .003
    final_z_tolerance_m: float = .004
    height_step_m: float = .002
    tracking_lag_m: float = .008
    feedback_gain: float = .2
    feedback_limit_m: float = .012
    feedback_period: int = 2
    stable_steps: int = 3
    table_margin_m: float = .004
    cube_margin_m: float = .0003
    segment_samples: int = 4
    check_cube: bool = True
    feedback: bool = True
    close_dwell_steps: int = 25

    def __post_init__(self) -> None:
        """Reject settings that silently disable progress or geometric sampling."""
        if self.segment_samples < 1 or self.feedback_period < 1 or self.height_step_m <= 0:
            raise ValueError("Positive segment sampling, feedback period and height step required")


def next_height(reference: float, measured: float, goal: float, cfg: DescentConfig) -> float:
    """Wait when the arm lags; otherwise advance a small vertical waypoint."""
    return reference if measured - reference > cfg.tracking_lag_m else max(goal, reference - cfg.height_step_m)


def tracking_bias(bias: np.ndarray, error: np.ndarray, cfg: DescentConfig) -> np.ndarray:
    """Bound Cartesian tracking compensation; never modify physical kp/kd."""
    return np.clip(bias - cfg.feedback_gain * error, -cfg.feedback_limit_m, cfg.feedback_limit_m)


class CartesianDescent(RaisedApproach):
    """Raise first, align high, then descend by checked Cartesian waypoints."""

    @classmethod
    def for_env(cls, env, descent: DescentConfig = DescentConfig()):
        """Build all private kinematic scratch data before wrapping the backend."""
        approach = ApproachConfig("cartesian", xy_tolerance_m=descent.xy_tolerance_m,
                                  z_tolerance_m=descent.final_z_tolerance_m)
        policy = super().for_env(env, approach)
        policy.descent = descent
        policy.options = replace(policy.options, grasp_height_m=descent.grasp_height_m)
        policy._cfg = replace(policy._cfg, grasp_height_m=descent.grasp_height_m)
        policy.checker = CommandPoseChecker(env.unwrapped, ResidualLimits())
        policy.cube_geom = env.unwrapped.backend.model.geom(env.unwrapped.config.object.geom).id
        return policy

    def reset(self) -> None:
        """Clear path state and rejection diagnostics for each placement."""
        super().reset()
        self.descent_stage = "initial"
        self.path_goal = None
        self.reference_z = None
        self.high_aligned = 0
        self.descent_calls = 0
        self.clearance_rejections = 0
        self.minimum_proposed_clearance = None
        self.closure_rejections = 0
        self.close_start_step = None

    def _enter(self, phase, obs):
        """Keep the established clock but use the close window to finish descent if needed."""
        if phase == Phase.DESCEND and not self.aborted and self.stage != "raise":
            self.path_goal = obs[self._layout.object_pos].astype(float).copy()
            self.path_goal[2] += self.options.grasp_height_m
            self.reference_z = obs[self._layout.object_pos][2] + self._cfg.pregrasp_height_m
            self.descent_stage = "align_high"
            return self._target
        if phase == Phase.CLOSE and not self.aborted:
            return replace(self._target, gripper_rad=np.deg2rad(self._gripper.open_deg))
        return super()._enter(phase, obs)

    def _closing_action(self, obs):
        """Defer closure decisions until after the current Cartesian update."""
        return self._track(obs, self._target)

    def _clearance(self, q: np.ndarray) -> tuple[bool, float]:
        """Sample the proposed joint segment; this is not swept-volume certification."""
        start = self._q_cmd(self._obs_for_check)
        minimum = self.descent.table_margin_m + 1.
        for fraction in np.linspace(0, 1, self.descent.segment_samples + 1)[1:]:
            command = start + fraction * (q - start)
            pose = self.checker.predict(command)
            minimum = min(minimum, pose.clearance_m)
            if pose.clearance_m < self.descent.table_margin_m:
                return False, minimum
            if self.descent.check_cube:
                gap = min(mujoco.mj_geomDistance(self.checker.model, self.checker.data, g,
                                                self.cube_geom, .05, None) for g in self.checker.geoms)
                if gap < self.descent.cube_margin_m:
                    return False, minimum
            for jaw in (self._gripper.close_deg, (self._gripper.close_deg + self._gripper.open_deg) / 2):
                envelope = command.copy()
                envelope[-1] = np.deg2rad(jaw)
                clearance = self.checker.predict(envelope).clearance_m
                minimum = min(minimum, clearance)
                if clearance < self.descent.table_margin_m:
                    return False, minimum
        return True, minimum

    def _target_for(self, waypoint: np.ndarray, obs: np.ndarray) -> None:
        """Accept only a converged target whose sampled command segment clears geometry."""
        result = self._solver.solve(waypoint + self.target_bias, self._q_cmd(obs), True)
        self.ik_misses += int(not result.converged())
        q = result.q.copy()
        q[:self._n_arm] = compensate(q[:self._n_arm], self.gravity.torque(q), self.kp,
                                    self._solver._arm_range, self.options.correction_limit_rad)
        self._obs_for_check = obs
        safe, minimum = self._clearance(q)
        self.minimum_proposed_clearance = minimum
        if safe and result.converged():
            self._target = replace(self._target, q_arm=q[:self._n_arm])
        else:
            # Includes nonconverged IK proposals, not only geometric rejections.
            self.clearance_rejections += 1

    def _update_path(self, obs: np.ndarray) -> None:
        """Align measured XY while high; keep the planned XY fixed during vertical descent."""
        measured = obs[self._layout.pinch_pos].astype(float)
        waypoint = self.path_goal.copy()
        waypoint[2] = self.reference_z
        error = measured - waypoint
        self.descent_calls += 1
        if self.descent_stage == "align_high":
            ready = np.linalg.norm(error[:2]) <= self.descent.xy_tolerance_m and measured[2] >= self.path_goal[2] + .03
            self.high_aligned = self.high_aligned + 1 if ready else 0
            if self.high_aligned >= self.descent.stable_steps:
                self.descent_stage = "descend"
        if self.descent.feedback and self.descent_calls % self.descent.feedback_period == 0:
            correction = error.copy()
            if np.linalg.norm(error[:2]) <= self.descent.xy_tolerance_m:
                correction[:2] = 0
            stalled = self.descent_stage == "descend" and error[2] > self.descent.tracking_lag_m
            if not (self.descent_stage == "at_goal" or stalled) or abs(error[2]) <= self.descent.final_z_tolerance_m:
                correction[2] = 0
            self.target_bias = tracking_bias(self.target_bias, correction, self.descent)
        if self.descent_stage == "descend":
            self.reference_z = next_height(self.reference_z, measured[2], self.path_goal[2], self.descent)
            waypoint[2] = self.reference_z
            if self.reference_z <= self.path_goal[2] + 1e-9:
                self.descent_stage = "at_goal"
        self._target_for(waypoint, obs)

    def _closure_clear(self, obs) -> bool:
        """Check measured posture at sampled closing-jaw configurations."""
        q = obs[self._layout.q].astype(float).copy()
        for jaw in (self._gripper.open_deg, self._gripper.close_deg,
                    (self._gripper.open_deg + self._gripper.close_deg) / 2):
            q[-1] = np.deg2rad(jaw)
            if self.checker.predict(q).clearance_m < self.descent.table_margin_m:
                self.closure_rejections += 1
                return False
        return True

    def __call__(self, obs):
        """Continue Cartesian descent with the jaw open, then use measured closure checks."""
        action = super().__call__(obs)
        if self._phase in (Phase.DESCEND, Phase.CLOSE) and not self.aborted and not self.close_started:
            self._update_path(obs)
            action = self._track(obs, self._target)
            if self._phase == Phase.CLOSE and self.descent_stage == "at_goal":
                if self._closure_clear(obs):
                    action = RaisedApproach._closing_action(self, obs)
                    if self.close_started and self.close_start_step is None:
                        self.close_start_step = self._step
        return action


class CartesianGuard(AcquisitionGuard):
    """Require a closing dwell as well as retained contact before executing lift."""

    def step(self, action):
        """Late closure must not cause lifting while the jaw is still sweeping inward."""
        start = self.controller.close_start_step
        waiting = start is not None and self.controller._step - start < self.controller.descent.close_dwell_steps
        if self.steps >= self.lift_start and waiting and not self.controller.aborted and not self.gate.released:
            raw = self.unwrapped
            self.gate.update(raw.grasp_state.grasped, may_release=False)
            command = np.zeros_like(action)
            error = np.deg2rad(raw.config.gripper.close_deg) - raw.grasp_state.robot.q_cmd[-1]
            command[-1] = np.clip(error / raw.config.task.delta_gripper_max_rad, -1, 1)
            self.steps += 1
            obs, reward, terminated, truncated, info = self.env.step(command)
            info.update(lift_released=False, grasp_abort=self.gate.aborted, closure_settling=True)
            return obs, reward, terminated, truncated, info
        return super().step(action)

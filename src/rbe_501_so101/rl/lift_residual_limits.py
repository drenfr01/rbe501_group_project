"""Opt-in lift-only residual authority with nominal kinematic proposal checks."""

from dataclasses import dataclass

import mujoco
import numpy as np

from rbe_501_so101.envs.model_indices import descendant_body_ids
from rbe_501_so101.config import ResidualConfig
from rbe_501_so101.rl.residual import ResidualPickLift


@dataclass(frozen=True)
class ResidualLimits:
    """Limits in normalized actions, radians and world meters; no gain changes."""

    scale: float = .05
    command_budget_rad: float = .02
    clearance_m: float = .002
    holding_drop_m: float = .0005
    distance_search_m: float = .05


def limited_residual(action: np.ndarray, phase: str, grasped: bool, cfg: ResidualLimits) -> np.ndarray:
    """Protect acquisition and the jaw; allow small arm corrections only during lift."""
    result = np.zeros_like(action, dtype=np.float32)
    if phase == "lift" and grasped:
        result[:-1] = cfg.scale * np.clip(action[:-1], -1, 1)
    return result


def accept_prediction(clearance: float, pinch_z: float, holding_floor: float, cfg: ResidualLimits) -> bool:
    """Check nominal command-pose clearance and the holding-height floor."""
    return bool(clearance >= cfg.clearance_m and pinch_z >= holding_floor)


@dataclass(frozen=True)
class PosePrediction:
    """A command-pose check, not a dynamics or swept-path safety guarantee."""

    clearance_m: float
    pinch_z: float


class CommandPoseChecker:
    """Use private MuJoCo data; never advance or mutate the physical environment."""

    def __init__(self, raw, cfg: ResidualLimits) -> None:
        self.raw, self.cfg = raw, cfg
        self.model = raw.backend.model
        self.data = mujoco.MjData(self.model)
        self.table = self.model.geom(raw.config.sim.table_top_geom).id
        bodies = descendant_body_ids(self.model, raw.config.sim.robot_root_body)
        self.geoms = [g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) in bodies
                      and self._collides(g)]
        self.offset = np.asarray(raw.config.gripper.pinch_point_b(raw.config.object.side_m))

    def _collides(self, geom: int) -> bool:
        """Exclude visual meshes disabled by the working collision fixes."""
        m, t = self.model, self.table
        return bool((m.geom_contype[geom] & m.geom_conaffinity[t]) or
                    (m.geom_contype[t] & m.geom_conaffinity[geom]))

    def predict(self, command: np.ndarray) -> PosePrediction:
        """Measure nearest collidable robot/top distance and nominal pinch height."""
        self.data.qpos[:] = self.raw.backend.data.qpos
        self.data.qpos[self.raw.backend.ids.qpos_adr] = command
        mujoco.mj_kinematics(self.model, self.data)
        distance = min((mujoco.mj_geomDistance(self.model, self.data, g, self.table,
                                             self.cfg.distance_search_m, None) for g in self.geoms),
                       default=self.cfg.distance_search_m)
        site = self.raw.backend.ids.tcp_site
        rotation = self.data.site_xmat[site].reshape(3, 3)
        pinch = self.data.site_xpos[site] + rotation @ self.offset
        return PosePrediction(float(distance), float(pinch[2]))


class LimitedLiftResidual(ResidualPickLift):
    """Keep the existing script/guard, restricting only the new learner's authority."""

    def __init__(self, env, controller, cfg: ResidualLimits = ResidualLimits()) -> None:
        super().__init__(env, controller, ResidualConfig(scale=cfg.scale))
        self.limits = cfg
        self.checker = CommandPoseChecker(self.unwrapped, cfg)
        self.reference = None
        self.ever_success = False
        self.counters = {}

    def reset(self, **kwargs):
        """Discard command reference, holding latch and intervention statistics."""
        self.reference, self.ever_success = None, False
        self.counters = dict.fromkeys(("active_steps", "applied_steps", "clearance_rejections",
                                      "holding_rejections", "budget_clips", "applied_norm_sum"), 0)
        return super().reset(**kwargs)

    def _commands(self, base: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bound the candidate around a separate rate-limited scripted command reference."""
        raw = self.unwrapped
        command = raw.grasp_state.robot.q_cmd.copy()
        scales = np.array([raw.config.task.delta_q_max_rad] * 5 + [raw.config.task.delta_gripper_max_rad])
        baseline = command + np.clip(base, -1, 1) * scales
        candidate = command + np.clip(base + residual, -1, 1) * scales
        budget = self.limits.command_budget_rad
        bounded = np.clip(candidate[:5], self.reference - budget, self.reference + budget)
        self.counters["budget_clips"] += int(np.any(np.abs(bounded - candidate[:5]) > 1e-8))
        candidate[:5] = bounded
        joint_limits = raw.backend.joint_limits()
        candidate[:5] = np.clip(candidate[:5], joint_limits.lower, joint_limits.upper)
        return baseline, candidate

    def _advance_reference(self) -> None:
        """Advance the shadow script even when a residual is zero or rejected."""
        if self.reference is None:
            self.reference = self.unwrapped.grasp_state.robot.q_cmd[:5].copy()
        target = self._controller._target
        self.reference += np.clip(target.q_arm - self.reference, -target.arm_rate_rad, target.arm_rate_rad)

    def _proposal(self, base: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """Reject a residual proposal, falling back to the existing scripted action."""
        baseline, candidate = self._commands(base, residual)
        nominal, predicted = self.checker.predict(baseline), self.checker.predict(candidate)
        floor = nominal.pinch_z - self.limits.holding_drop_m if self.ever_success else -np.inf
        clear = predicted.clearance_m >= self.limits.clearance_m
        holding = predicted.pinch_z >= floor
        self.counters["clearance_rejections"] += int(not clear)
        self.counters["holding_rejections"] += int(not holding)
        if not accept_prediction(predicted.clearance_m, predicted.pinch_z, floor, self.limits):
            return base
        self.counters["applied_steps"] += 1
        raw = self.unwrapped
        command = raw.grasp_state.robot.q_cmd
        scales = np.array([raw.config.task.delta_q_max_rad] * 5 + [raw.config.task.delta_gripper_max_rad])
        return np.clip((candidate - command) / scales, -1, 1).astype(np.float32)

    def step(self, action):
        """Apply only allowed corrections; report interventions separately from success."""
        base = self.base_action()
        raw = self.unwrapped
        phase = str(self._controller.phase)
        if phase == "lift":
            self._advance_reference()
        gate = self.env.env.gate  # ContactAudit -> GuardedLift.
        released = gate.released and not gate.aborted
        residual = limited_residual(np.asarray(action), phase, raw.grasp_state.grasped and released, self.limits)
        executed = base
        if np.any(residual):
            self.counters["active_steps"] += 1
            executed = self._proposal(base, residual)
        self.counters["applied_norm_sum"] += float(np.linalg.norm(executed - base))
        obs, reward, terminated, truncated, info = self.env.step(executed)
        self._obs, self._base = obs, None
        self.last_executed_action = executed.copy()
        self.ever_success |= bool(info["is_success"])
        info.update(self.counters)
        return obs, reward, terminated, truncated, info

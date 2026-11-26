"""Full destination-conditioned pick-and-place Gym task with bounded scripted residuals."""

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rbe_501_so101.control.carry_place import TransferPhase, destination
from rbe_501_so101.control.contact_tracked_place import ContactTrackedCarryPlace
from rbe_501_so101.control.pick_place_stepper import PickPlaceStepper
from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
from rbe_501_so101.envs.pick_place_config import (
    PickPlaceConfig, PickPlaceObservationLayout, reward_terms,
)
from rbe_501_so101.envs.protocols import ObjectPlacer


@dataclass
class YawOffsetPlacer:
    """Preserve the old center/mass RNG stream while exposing independent cube yaw."""

    base: ObjectPlacer
    yaw_rad: float = 0.

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        """Add the explicitly sampled orientation without consuming extra placement draws."""
        center, yaw = self.base.sample(rng)
        return center, yaw + self.yaw_rad


class SO101PickPlaceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Complete-task residual environment; zero actions reproduce the validated scripted baseline."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg: PickPlaceConfig = PickPlaceConfig(), render_mode=None) -> None:
        """Construct a separate environment; do not modify existing Gym registrations or models."""
        self.config, self.render_mode, self.layout = cfg, render_mode, PickPlaceObservationLayout()
        self.inner = make_timed_env(cfg.preset, cfg.timing, render_mode)
        raw = self.inner.unwrapped
        self.placer = YawOffsetPlacer(raw._placer)
        raw._placer = self.placer
        self.action_space = spaces.Box(-1., 1., (3,), np.float32)
        bound = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-bound, bound, (self.layout.size,), np.float32)
        self.sequence = self.stepper = None
        self.steps = 0
        self.last_action = np.zeros(3)
        self.done = True

    def reset(self, *, seed=None, options=None):
        """Reset the full task; optional destination_xy is validated on the supported table workspace."""
        super().reset(seed=seed)
        self.done = True
        options = options or {}
        yaw = float(options.get("yaw_offset_deg", self.np_random.choice(self.config.yaw_offsets_deg)))
        if not np.isfinite(yaw):
            raise ValueError("Finite yaw offset required")
        self.placer.yaw_rad = float(np.deg2rad(yaw))
        self.inner.reset(seed=seed)
        self.sequence = ContactTrackedCarryPlace(self.inner, self.config.transfer, self.config.support, self.config.tracking)
        state = self.sequence.physical.grasp_state()
        self.sequence.target = destination(state.obj.p_s, self.sequence.backend.pan_axis_xy,
                                           self.sequence.backend.rest_z, self.config.transfer)
        if "destination_xy" in options:
            self.set_destination(options["destination_xy"])
        self.stepper = PickPlaceStepper(self.sequence, self.config)
        self.steps, self.done = 0, False
        self.last_action[:] = 0
        return self.observation(), self.info()

    def set_destination(self, xy) -> None:
        """Accept a finite resting point within the baseline's radial/azimuth workspace."""
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError("destination_xy must contain two finite coordinates")
        offset = xy - self.sequence.backend.pan_axis_xy
        placement = self.sequence.raw.config.placement
        radius, angle = np.linalg.norm(offset), abs(float(np.rad2deg(np.arctan2(offset[1], offset[0]))))
        if not placement.radial_min_m <= radius <= placement.radial_max_m or angle > placement.azimuth_max_deg:
            raise ValueError("Destination is outside the supported table workspace")
        self.sequence.target[:] = [*xy, self.sequence.backend.rest_z]

    def step(self, action):
        """Advance one complete-task control period and distinguish success/failure from timeout."""
        if self.done:
            raise RuntimeError("Call reset before stepping a finished pick/place episode")
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected three finite Cartesian residual actions")
        before = self.potential()
        self.last_action[:] = np.clip(action, -1, 1)
        self.stepper.tick(self.last_action)
        self.steps += 1
        terminated = self.stepper.finished
        truncated = self.steps >= self.config.max_episode_steps and not terminated
        self.done = terminated or truncated
        terms = reward_terms(before, self.potential(), self.stepper.applied_action, self.stepper.success,
                             bool(self.sequence.failure), self.config.reward)
        info = self.info()
        info.update({f"reward_{k}": v for k, v in asdict(terms).items()})
        if self.render_mode == "human" and self.stepper.phase != TransferPhase.ACQUIRE:
            self.inner.unwrapped.render()
        return self.observation(), terms.total, terminated, truncated, info

    def potential(self) -> float:
        """Stage-specific distance potential; final success remains independent physical evidence."""
        state = self.sequence.backend.grasp_state()
        phase = self.stepper.phase
        if phase == TransferPhase.ACQUIRE:
            distance_m = np.linalg.norm(state.obj.p_s - state.p_pinch)
        elif phase == TransferPhase.CARRY:
            distance_m = np.linalg.norm(self.stepper.carry_goal - state.obj.p_s)
        elif phase == TransferPhase.RETREAT:
            height = state.p_pinch[2] - self.stepper.retreat_start[2]
            distance_m = max(0., self.config.transfer.retreat_clearance_m - height)
        else:
            distance_m = np.linalg.norm(self.sequence.target - state.obj.p_s)
        return -self.config.reward.distance_scale * float(distance_m)

    def observation(self) -> np.ndarray:
        """Expose cube orientation/velocity, destination and phase/controller state to the policy."""
        state = self.sequence.backend.grasp_state()
        model, data = self.sequence.backend.model, self.sequence.backend.data
        joint = model.body_jntadr[self.sequence.cube]
        adr = model.jnt_qposadr[joint]
        reference = self.sequence.reference
        solver = self.sequence.solver or self.inner._controller._solver
        axis = getattr(solver, "desired_axis", None)
        phases = list(TransferPhase)
        fsm = self.stepper
        retreat = np.zeros(3) if fsm.retreat_goal is None else fsm.retreat_goal - state.p_pinch
        progress = [fsm.phase_steps / fsm.budget, self.steps / self.config.max_episode_steps,
                    fsm.stable / self.config.transfer.stable_steps]
        blocks = [state.robot.qpos, state.robot.qvel, state.robot.q_cmd - state.robot.qpos,
                  state.p_pinch, state.obj.p_s, data.qpos[adr + 3:adr + 7],
                  data.qvel[self.sequence.cube_dof:self.sequence.cube_dof + 6], self.sequence.target,
                  self.sequence.target - state.obj.p_s, self.last_action,
                  [float(fsm.phase == p) for p in phases], progress,
                  [state.grasped, self.sequence.contacts()[0]], [state.obj.mass_kg],
                  np.zeros(3) if reference is None else reference - state.p_pinch,
                  self.sequence.tracking_bias, np.zeros(3) if axis is None else axis,
                  [0. if fsm.carry_goal is None else fsm.carry_goal[2]], retreat, [reference is not None],
                  [fsm.lifted, fsm.released, fsm.retreated]]
        return np.concatenate([np.asarray(b).ravel() for b in blocks]).astype(np.float32)

    def info(self) -> dict:
        """Report placement outcome, safety events and actual residual authority separately."""
        status = self.stepper.status()
        return {"is_success": self.stepper.success, "phase": str(self.stepper.phase),
                "failure_reason": self.sequence.failure or "", "destination": self.sequence.target.copy(),
                "placement_error_m": status.xy_error_m, "height_error_m": status.height_error_m,
                "grasped": status.grasped, "supported": status.supported,
                "lifted": status.lifted, "released": status.released, "retreated": status.retreated,
                "mass_kg": self.sequence.physical.grasp_state().obj.mass_kg,
                "robot_table_contact_steps": sum(f.table_contact for f in self.sequence.frames),
                "mujoco_warning": any(f.warning for f in self.sequence.frames),
                "residual_steps": self.stepper.residual_steps, "residual_rejections": self.stepper.residual_rejections,
                "residual_applied": bool(np.any(self.stepper.applied_action)),
                "residual_norm_m": float(np.linalg.norm(self.stepper.applied_action) * self.config.residual_scale_m),
                "angular_speed_rad_s": status.angular_speed_rad_s, "episode_steps": self.steps}

    def render(self):
        """Delegate rendering to the original calibrated backend."""
        return self.inner.unwrapped.render()

    def close(self) -> None:
        """Release the original MuJoCo backend and viewer resources."""
        self.inner.close()

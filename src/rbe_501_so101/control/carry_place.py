"""Experimental backend-level carry/place after the unchanged 200-step acquisition."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from rbe_501_so101.config import PinchIKConfig
from rbe_501_so101.control.aligned_pick_lift import compensate
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.envs.model_indices import descendant_body_ids
from rbe_501_so101.kinematics.oracle_jaw_ik import JawIKSolver


@dataclass(frozen=True)
class TransferConfig:
    """Small first demonstration: meters, radians and real 33 ms control periods."""

    destination_angle_deg: float = 12.
    cartesian_step_m: float = .002
    tracking_lag_m: float = .012
    arm_rate_rad: float = .02
    open_rate_rad: float = .025
    xy_tolerance_m: float = .005
    support_height_m: float = .005
    speed_tolerance_m_s: float = .01
    carry_tolerance_m: float = .004
    final_tolerance_m: float = .01
    retreat_m: float = .05
    retreat_clearance_m: float = .04
    stable_steps: int = 5
    carry_steps: int = 150
    lower_steps: int = 150
    release_steps: int = 60
    retreat_steps: int = 100
    settle_steps: int = 60
    clearance_m: float = .004


class TransferPhase(StrEnum):
    """Separate intentional release from unintentional grasp loss during transport."""

    ACQUIRE = "acquire"
    CARRY = "carry"
    LOWER = "lower"
    RELEASE = "release"
    RETREAT = "retreat"
    SETTLE = "settle"


def destination(source: np.ndarray, axis_xy: np.ndarray, rest_z: float, cfg: TransferConfig) -> np.ndarray:
    """Choose a nearby point at the same radius, rotating toward the workspace center."""
    offset = source[:2] - axis_xy
    angle = float(np.arctan2(offset[1], offset[0]))
    angle += np.deg2rad(cfg.destination_angle_deg) * (-1 if angle >= 0 else 1)
    xy = axis_xy + np.linalg.norm(offset) * np.array([np.cos(angle), np.sin(angle)])
    return np.array([*xy, rest_z])


def release_ready(supported: bool, xy_error: float, height: float, speed: float, cfg: TransferConfig) -> bool:
    """Authorize opening only after a low, slow cube is supported at its destination."""
    return (supported and xy_error <= cfg.xy_tolerance_m and abs(height) <= cfg.support_height_m
            and speed <= cfg.speed_tolerance_m_s)


@dataclass(frozen=True)
class TransferFrame:
    """Physical cube/hand state and the phase-specific contact outcome."""

    phase: str
    step: int
    cube_m: tuple[float, float, float]
    pinch_m: tuple[float, float, float]
    grasped: bool
    supported: bool
    speed_m_s: float
    jaw_deg: float
    table_contact: bool
    warning: bool
    qpos: tuple[float, ...]


class CarryPlace:
    """Keep acquisition intact; then command the same calibrated backend with new guards."""

    def __init__(self, env, cfg: TransferConfig = TransferConfig()) -> None:
        """Own transfer state without extending or stepping a terminated Gym episode."""
        self.env, self.raw, self.cfg = env, env.unwrapped, cfg
        self.backend = self.raw.backend
        self.physical = self.backend.inner if isinstance(self.backend, GraspServoBackend) else self.backend
        self.robot_bodies = descendant_body_ids(self.backend.model, self.raw.config.sim.robot_root_body)
        self.table = self.backend.model.body(self.raw.config.sim.table_body).id
        self.cube = self.backend.model.body(self.raw.config.object.body).id
        joint = self.backend.model.body_jntadr[self.cube]
        self.cube_dof = self.backend.model.jnt_dofadr[joint]
        self.frames: list[TransferFrame] = []
        self.failure: str | None = None
        self.target: np.ndarray | None = None
        self.solver = None
        self.reference = None

    def contacts(self) -> tuple[bool, bool]:
        """Return cube/table support and any robot/table collision independently."""
        supported = robot_table = False
        model = self.backend.model
        for contact in self.backend.data.contact:
            pair = {int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
            supported |= self.table in pair and self.cube in pair
            robot_table |= self.table in pair and bool(pair & self.robot_bodies)
        return supported, robot_table

    def record(self, phase: TransferPhase) -> TransferFrame:
        """Read true physical poses; controllers still receive the original sensed joints."""
        state = self.physical.grasp_state()
        supported, table_hit = self.contacts()
        speed = float(np.linalg.norm(self.backend.data.qvel[self.cube_dof:self.cube_dof + 3]))
        frame = TransferFrame(str(phase), len(self.frames) + 1, tuple(state.obj.p_s), tuple(state.p_pinch),
                              state.grasped, supported, speed, float(np.rad2deg(state.robot.qpos[-1])),
                              table_hit, self.backend.has_warnings(), tuple(self.backend.data.qpos))
        self.frames.append(frame)
        if table_hit or frame.warning:
            self.failure = "robot_table_contact" if table_hit else "mujoco_warning"
        return frame

    def acquire(self, seed: int) -> bool:
        """Run the validated Gym task to its normal truncation, then begin a separate rollout."""
        self.env.reset(seed=seed)
        initial = self.physical.grasp_state().obj.p_s.copy()
        self.target = destination(initial, self.backend.pan_axis_xy, self.backend.rest_z, self.cfg)
        for _ in range(self.raw.config.task.max_episode_steps):
            _, _, _, _, info = self.env.step(np.zeros(6))
            self.record(TransferPhase.ACQUIRE)
        if not info["is_success"] or self.failure:
            self.failure = self.failure or "acquisition_failed"
            return False
        self._transfer_solver()
        return True

    def _transfer_solver(self) -> None:
        """Maintain a horizontal grasp axis during transport with the existing position-first IK."""
        base = self.env._controller._solver
        self.solver = JawIKSolver(base._model, base._ids, PinchIKConfig(), base._p_b, lambda: 0.)
        _, rotation = self.solver.tcp_pose(self.backend.grasp_state().robot.qpos)
        axis = rotation[:, 0].copy()
        axis[2] = 0
        self.solver.desired_axis = axis / np.linalg.norm(axis)

    def move(self, goal: np.ndarray, phase: TransferPhase) -> bool:
        """Command one checked small Cartesian step, retaining the separate closed-jaw command."""
        if phase in (TransferPhase.CARRY, TransferPhase.LOWER) and not self.physical.grasp_state().grasped:
            self.failure = "grasp_lost_during_transport"
            return False
        state = self.backend.grasp_state()
        if self.reference is None:
            self.reference = state.p_pinch.copy()
        step = goal - self.reference
        if np.linalg.norm(self.reference - state.p_pinch) < self.cfg.tracking_lag_m:
            self.reference += step * min(1., self.cfg.cartesian_step_m / max(np.linalg.norm(step), 1e-12))
        waypoint = self.reference
        result = self.solver.solve(waypoint, state.robot.qpos, True)
        if not result.converged():
            self.failure = "transfer_ik_failed"
            return False
        controller = self.env._controller
        arm = compensate(result.q[:5], controller.gravity.torque(result.q), controller.kp,
                         self.solver._arm_range, controller.options.correction_limit_rad)
        command = state.robot.q_cmd.copy()
        command[:5] += np.clip(arm - command[:5], -self.cfg.arm_rate_rad, self.cfg.arm_rate_rad)
        if not self.safe_command(state.robot.q_cmd, command):
            self.failure = "transfer_clearance_rejected"
            return False
        self.backend.apply_delta(command[:5] - state.robot.q_cmd[:5])
        return self.advance(phase)

    def safe_command(self, start: np.ndarray, command: np.ndarray) -> bool:
        """Sample the short joint command segment using the existing collidable-geometry checker."""
        samples = self.env._controller.descent.segment_samples
        return all(self.env.checker.predict(start + t * (command - start)).clearance_m >= self.cfg.clearance_m
                   for t in np.linspace(0, 1, samples + 1)[1:])

    def advance(self, phase: TransferPhase) -> bool:
        """Advance existing servo physics and stop transport on unexpected grasp loss."""
        if phase in (TransferPhase.CARRY, TransferPhase.LOWER) and not self.physical.grasp_state().grasped:
            self.failure = "grasp_lost_during_transport"
            return False
        self.backend.step()
        frame = self.record(phase)
        if phase in (TransferPhase.CARRY, TransferPhase.LOWER) and not frame.grasped:
            self.failure = "grasp_lost_during_transport"
        return self.failure is None

    def move_cube(self, goal: np.ndarray, phase: TransferPhase, budget: int) -> bool:
        """Feedback on actual cube displacement instead of assuming a perfectly rigid grasp offset."""
        stable = 0
        self.reference = None
        for _ in range(budget):
            sensed = self.backend.grasp_state()
            cube = self.physical.grasp_state().obj.p_s
            error = goal - cube
            if phase == TransferPhase.LOWER:
                ready = self.support_ready()
            else:
                ready = np.linalg.norm(error) <= self.cfg.carry_tolerance_m
            stable = stable + 1 if ready else 0
            if stable >= self.cfg.stable_steps:
                return True
            if not self.move(sensed.p_pinch + error, phase):
                return False
        self.failure = f"{phase}_timeout"
        return False

    def support_ready(self) -> bool:
        """Check actual table contact, target error, resting height and translational speed."""
        state = self.physical.grasp_state()
        speed = float(np.linalg.norm(self.backend.data.qvel[self.cube_dof:self.cube_dof + 3]))
        xy = float(np.linalg.norm(state.obj.p_s[:2] - self.target[:2]))
        return release_ready(self.contacts()[0], xy, state.obj.p_s[2] - self.backend.rest_z, speed, self.cfg)

    def release(self) -> bool:
        """Open slowly with the arm held; contact loss is intentional only after support authorization."""
        if not self.support_ready():
            self.failure = "release_without_support"
            return False
        for _ in range(self.cfg.release_steps):
            state = self.backend.grasp_state()
            error = np.deg2rad(self.raw.config.gripper.open_deg) - state.robot.q_cmd[-1]
            delta = float(np.clip(error, -self.cfg.open_rate_rad, self.cfg.open_rate_rad))
            proposed = state.robot.q_cmd.copy()
            proposed[-1] += delta
            if not self.safe_command(state.robot.q_cmd, proposed):
                self.failure = "release_clearance_rejected"
                return False
            self.backend.apply_gripper_delta(delta)
            if not self.advance(TransferPhase.RELEASE):
                return False
        return True

    def retreat(self) -> bool:
        """Move the open gripper upward, keeping final placement separate from continued holding."""
        start = self.backend.grasp_state().p_pinch.copy()
        goal = start + [0, 0, self.cfg.retreat_m]
        self.reference = None
        for _ in range(self.cfg.retreat_steps):
            if self.backend.grasp_state().p_pinch[2] - start[2] >= self.cfg.retreat_clearance_m:
                return True
            if not self.move(goal, TransferPhase.RETREAT):
                return False
        self.failure = "retreat_timeout"
        return False

    def run_transfer(self) -> bool:
        """Carry high, lower to supported rest, release, retreat and verify persistent placement."""
        carry = self.target.copy()
        carry[2] = self.physical.grasp_state().obj.p_s[2]
        if not self.move_cube(carry, TransferPhase.CARRY, self.cfg.carry_steps):
            return False
        if not self.move_cube(self.target, TransferPhase.LOWER, self.cfg.lower_steps):
            return False
        if not self.release() or not self.retreat():
            return False
        for _ in range(self.cfg.settle_steps):
            if not self.advance(TransferPhase.SETTLE):
                return False
        cube = self.physical.grasp_state().obj.p_s
        last = self.frames[-1]
        success = (last.supported and not last.grasped and last.speed_m_s <= self.cfg.speed_tolerance_m_s
                   and abs(cube[2] - self.backend.rest_z) <= self.cfg.support_height_m
                   and np.linalg.norm(cube[:2] - self.target[:2]) <= self.cfg.final_tolerance_m)
        self.failure = None if success else "final_placement_failed"
        return success

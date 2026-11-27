"""One-control-period state machine reusing the validated complete-task primitives."""

import numpy as np

from rbe_501_so101.control.carry_place import TransferPhase
from rbe_501_so101.envs.pick_place_config import PickPlaceConfig, PlacementStatus, placement_success


class PickPlaceStepper:
    """Expose each physical control period to RL instead of running blocking phase loops."""

    def __init__(self, sequence, cfg: PickPlaceConfig) -> None:
        """Own phase/time state while leaving the validated controller implementation intact."""
        self.sequence, self.cfg = sequence, cfg
        self.phase = TransferPhase.ACQUIRE
        self.phase_steps = self.stable = 0
        self.finished = self.success = False
        self.lifted = self.released = self.retreated = self.settled = False
        self.carry_goal = self.retreat_goal = self.retreat_start = None
        self.applied_action = np.zeros(3)
        self.residual_rejections = self.residual_steps = 0

    @property
    def budget(self) -> int:
        """The same per-phase budgets as the original complete-task sequence."""
        if self.phase == TransferPhase.ACQUIRE:
            return self.sequence.raw.config.task.max_episode_steps
        return getattr(self.cfg.transfer, f"{self.phase}_steps")

    def enter(self, phase: TransferPhase) -> None:
        """Reset only the phase-local reference and readiness counters, as the blocking script does."""
        self.phase, self.phase_steps, self.stable = phase, 0, 0
        self.sequence.reference = None
        if phase == TransferPhase.CARRY:
            self.carry_goal = self.sequence.target.copy()
            self.carry_goal[2] = self.sequence.physical.grasp_state().obj.p_s[2]
        if phase == TransferPhase.LOWER:
            self.sequence.tracking_bias[:] = 0
        if phase == TransferPhase.RETREAT:
            self.retreat_start = self.sequence.backend.grasp_state().p_pinch.copy()
            self.retreat_goal = self.retreat_start + [0, 0, self.cfg.transfer.retreat_m]

    def fail(self, reason: str) -> None:
        """Terminate on a real controller/safety failure rather than mislabeling a timeout success."""
        self.sequence.failure, self.finished = reason, True

    def status(self) -> PlacementStatus:
        """Measure physical completion criteria independently of the reward."""
        state = self.sequence.physical.grasp_state()
        speed = float(np.linalg.norm(self.sequence.backend.data.qvel[self.sequence.cube_dof:self.sequence.cube_dof + 3]))
        return PlacementStatus(self.lifted, self.released, self.retreated, self.settled,
                               self.sequence.contacts()[0], state.grasped,
                               float(np.linalg.norm(state.obj.p_s[:2] - self.sequence.target[:2])),
                               float(state.obj.p_s[2] - self.sequence.backend.rest_z), speed,
                               float(np.linalg.norm(self.sequence.backend.data.qvel[self.sequence.cube_dof + 3:self.sequence.cube_dof + 6])))

    def tick(self, action: np.ndarray) -> None:
        """Advance exactly one physics control period, apart from a terminal precheck rejection."""
        before = len(self.sequence.frames)
        self.applied_action[:] = 0
        while len(self.sequence.frames) == before and not self.finished:
            if self.phase == TransferPhase.ACQUIRE:
                self.acquire()
            elif self.phase in (TransferPhase.CARRY, TransferPhase.LOWER):
                self.transport(action)
            elif self.phase == TransferPhase.RELEASE:
                self.open_gripper()
            elif self.phase == TransferPhase.RETREAT:
                self.retreat()
            else:
                self.settle()
        if self.sequence.failure:
            self.finished = True

    def acquire(self) -> None:
        """Use the existing guarded pickup for its original 200 periods only."""
        _, _, _, _, info = self.sequence.env.step(np.zeros(6))
        self.sequence.record(TransferPhase.ACQUIRE)
        self.phase_steps += 1
        if self.phase_steps == self.budget:
            if not info["is_success"] or self.sequence.failure:
                return self.fail(self.sequence.failure or "acquisition_failed")
            self.lifted = True
            self.sequence._transfer_solver()
            self.enter(TransferPhase.CARRY)

    def transport(self, action: np.ndarray) -> None:
        """Keep measured support authorization and all phase budgets identical to the script."""
        sensed = self.sequence.backend.grasp_state()
        cube = self.sequence.physical.grasp_state().obj.p_s
        goal = self.carry_goal.copy() if self.phase == TransferPhase.CARRY else self.sequence.target.copy()
        if self.phase == TransferPhase.LOWER:
            goal[2] -= self.cfg.support.maximum_descent_m
        error = goal - cube
        ready = np.linalg.norm(error) <= self.cfg.transfer.carry_tolerance_m if self.phase == TransferPhase.CARRY else self.sequence.support_ready()
        self.stable = self.stable + 1 if ready else 0
        if self.stable >= self.cfg.transfer.stable_steps:
            next_phase = TransferPhase.LOWER if self.phase == TransferPhase.CARRY else TransferPhase.RELEASE
            self.enter(next_phase)
            return
        self.motion(sensed.p_pinch + error, action)
        self.phase_steps += 1
        if not self.sequence.failure and self.phase_steps >= self.budget:
            self.fail(f"{self.phase}_timeout")

    def motion(self, goal: np.ndarray, action: np.ndarray) -> None:
        """Try a <=1mm Cartesian correction and fall back if its kinematic proposal is rejected."""
        residual = np.asarray(action, dtype=float).copy()
        if not self.sequence.physical.grasp_state().grasped:
            residual[:] = 0
        if self.phase == TransferPhase.LOWER and self.sequence.contacts()[0]:
            residual[2] = 0
        residual /= max(1., np.linalg.norm(residual))
        reference = None if self.sequence.reference is None else self.sequence.reference.copy()
        bias, count = self.sequence.tracking_bias.copy(), len(self.sequence.frames)
        ok = self.sequence.move(goal + self.cfg.residual_scale_m * residual, self.phase)
        rejectable = self.sequence.failure in ("transfer_ik_failed", "transfer_clearance_rejected")
        if not ok and rejectable and len(self.sequence.frames) == count and np.any(residual):
            self.sequence.reference, self.sequence.tracking_bias, self.sequence.failure = reference, bias, None
            self.residual_rejections += 1
            self.sequence.move(goal, self.phase)
            return
        if not ok and len(self.sequence.frames) == count:
            return
        self.applied_action[:] = residual
        self.residual_steps += int(np.any(residual))

    def open_gripper(self) -> None:
        """Protect the jaw: the policy cannot release before stable measured table support."""
        if self.phase_steps == 0 and not self.sequence.support_ready():
            return self.fail("release_without_support")
        state = self.sequence.backend.grasp_state()
        error = np.deg2rad(self.sequence.raw.config.gripper.open_deg) - state.robot.q_cmd[-1]
        delta = float(np.clip(error, -self.cfg.transfer.open_rate_rad, self.cfg.transfer.open_rate_rad))
        proposed = state.robot.q_cmd.copy()
        proposed[-1] += delta
        if not self.sequence.safe_command(state.robot.q_cmd, proposed):
            return self.fail("release_clearance_rejected")
        self.sequence.backend.apply_gripper_delta(delta)
        self.sequence.advance(TransferPhase.RELEASE)
        self.phase_steps += 1
        if not self.sequence.failure and self.phase_steps == self.budget:
            self.released = True
            self.enter(TransferPhase.RETREAT)

    def retreat(self) -> None:
        """Require the same measured hand clearance instead of accepting an object still held."""
        measured = self.sequence.backend.grasp_state().p_pinch
        if measured[2] - self.retreat_start[2] >= self.cfg.transfer.retreat_clearance_m:
            self.retreated = True
            self.enter(TransferPhase.SETTLE)
            return
        self.sequence.move(self.retreat_goal, TransferPhase.RETREAT)
        self.phase_steps += 1
        if not self.sequence.failure and self.phase_steps >= self.budget:
            self.fail("retreat_timeout")

    def settle(self) -> None:
        """Finish only after the full final observation window and physical placement criteria."""
        self.sequence.advance(TransferPhase.SETTLE)
        self.phase_steps += 1
        if not self.sequence.failure and self.phase_steps == self.budget:
            self.settled, self.finished = True, True
            self.success = placement_success(self.status(), self.cfg)
            if not self.success:
                self.sequence.failure = "final_placement_failed"

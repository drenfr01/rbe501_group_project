"""Opt-in bounded acquisition settling; original oracle and baseline remain frozen."""

from dataclasses import dataclass, replace

import numpy as np

from rbe_501_so101.control.cartesian_descent import CartesianGuard
from rbe_501_so101.control.oracle_acquisition import OracleAcquisition
from rbe_501_so101.control.scripted_pick_lift import Phase
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from rbe_501_so101.kinematics.oracle_jaw_ik import JawIKConfig, JawIKSolver
from rbe_501_so101.rl.calibrated_pick_lift import ContactAudit
from rbe_501_so101.rl.cartesian_zero_training import CartesianLiftResidual, CartesianZeroPreset


@dataclass(frozen=True)
class TimingConfig:
    """Real control-step deadlines within the original 200-step episode."""

    alignment_deadline_step: int = 120
    lift_release_deadline_step: int = 150
    minimum_lift_steps: int = 50

    def __post_init__(self) -> None:
        """Reject reversed deadlines and nonpositive time budgets."""
        if not 0 < self.alignment_deadline_step < self.lift_release_deadline_step or self.minimum_lift_steps < 1:
            raise ValueError("Ordered positive acquisition deadlines and lift budget required")


def settling_phase(step: int, nominal: Phase, close_start: int | None, dwell: int, cfg: TimingConfig) -> Phase:
    """Delay nominal lift for readiness and real elapsed closing dwell, with a hard bound."""
    if nominal != Phase.LIFT:
        return nominal
    if close_start is not None:
        return Phase.CLOSE if step < close_start + dwell else Phase.LIFT
    return Phase.CLOSE if step < cfg.alignment_deadline_step else Phase.LIFT


class TimedOracleAcquisition(OracleAcquisition):
    """Keep real time monotonic while permitting bounded measured acquisition settling."""

    timing = TimingConfig()

    def _phase_at(self, step: int) -> Phase:
        """Do not enter lift until closure has settled, or the alignment deadline expires."""
        nominal = super()._phase_at(step)
        return settling_phase(step, nominal, self.close_start_step, self.descent.close_dwell_steps, self.timing)


class TimedCartesianGuard(CartesianGuard):
    """Follow the controller's actual phase without allowing contact-free or endless lift waits."""

    def step(self, action):
        """Preserve all old contact/dwell checks; use the actual lift transition for gating."""
        controller = self.controller
        late = self.steps >= controller.timing.lift_release_deadline_step and not self.gate.released
        if late and not controller.aborted:
            controller.aborted, controller.abort_reason = True, "grasp_deadline"
        self.lift_start = self.steps + int(controller.phase != Phase.LIFT)
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(acquisition_phase=str(controller.phase), acquisition_step=controller._step,
                    acquisition_abort_reason=controller.abort_reason)
        return obs, reward, terminated, truncated, info


def make_timed_env(preset: CartesianZeroPreset, timing: TimingConfig = TimingConfig(), render_mode=None):
    """Construct a separate timed candidate without editing either prior factory."""
    cfg = preset.environment()
    if timing.lift_release_deadline_step + timing.minimum_lift_steps > cfg.task.max_episode_steps:
        raise ValueError("Timing must reserve the minimum lift budget inside the episode")
    if timing.alignment_deadline_step + preset.resolved_descent().close_dwell_steps >= timing.lift_release_deadline_step:
        raise ValueError("Closing dwell must fit before the lift-release deadline")
    raw = SO101PickLiftEnv(cfg=cfg, render_mode=render_mode)
    controller = TimedOracleAcquisition.for_env(raw, preset.resolved_descent())
    original = controller._solver
    body = raw.backend.model.body(raw.config.object.body).id

    def cube_yaw() -> float:
        """Use the same privileged initial-orientation input as the frozen oracle."""
        rotation = raw.backend.data.xmat[body].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    controller._solver = JawIKSolver(original._model, original._ids, original._cfg,
                                    original._p_b, cube_yaw, JawIKConfig())
    controller.timing, controller.options = timing, preset.base.motion
    controller._cfg = replace(controller._cfg, close_rate_rad=preset.closing.rate_rad)
    if preset.base.arm_effects:
        raw.backend = GraspServoBackend(raw.backend, raw.config)
    guard = TimedCartesianGuard(raw, controller)
    guard.stable_steps = preset.base.stable_steps
    return CartesianLiftResidual(ContactAudit(guard), controller, preset.limits)

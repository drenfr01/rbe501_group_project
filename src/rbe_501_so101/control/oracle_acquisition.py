"""Opt-in Cartesian acquisition with privileged simulated cube orientation."""

from dataclasses import replace

import numpy as np

from rbe_501_so101.control.cartesian_descent import CartesianDescent, CartesianGuard
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from rbe_501_so101.kinematics.oracle_jaw_ik import JawIKConfig, JawIKSolver
from rbe_501_so101.rl.calibrated_pick_lift import ContactAudit
from rbe_501_so101.rl.cartesian_zero_training import CartesianLiftResidual, CartesianZeroPreset


class OracleAcquisition(CartesianDescent):
    """Rotate the jaw axis toward a cube face while preserving acquisition safeguards."""

    def reset(self) -> None:
        """Discard the previous episode's oracle axis and measured alignment diagnostics."""
        super().reset()
        self.jaw_alignment_rad = None
        if isinstance(self._solver, JawIKSolver):
            self._solver.reset_axis()

    def _align(self, goal, q):
        """The experimental solver already aligns the jaw; do not apply radial reseeding."""
        return self._solver.solve(goal, q, True)

    def _alignment(self, obs: np.ndarray) -> bool:
        """Require actual sensed jaw alignment as well as the original position/tilt gate."""
        ready = super()._alignment(obs)
        _, rotation = self._solver.tcp_pose(obs[self._layout.q].astype(float))
        self.jaw_alignment_rad = float(np.arccos(np.clip(rotation[:, 0] @ self._solver.desired_axis, -1, 1)))
        return ready and self.jaw_alignment_rad <= self._solver.jaw_cfg.axis_tolerance_rad


def make_oracle_env(preset: CartesianZeroPreset, jaw_cfg: JawIKConfig = JawIKConfig(), render_mode=None):
    """Build a separate controller; keep model, gains, task, gripper and old presets intact."""
    raw = SO101PickLiftEnv(cfg=preset.environment(), render_mode=render_mode)
    controller = OracleAcquisition.for_env(raw, preset.resolved_descent())
    original = controller._solver
    body = raw.backend.model.body(raw.config.object.body).id

    def cube_yaw() -> float:
        """Read true MuJoCo cube rotation, explicitly unavailable to the saved policy."""
        rotation = raw.backend.data.xmat[body].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    controller._solver = JawIKSolver(original._model, original._ids, original._cfg,
                                    original._p_b, cube_yaw, jaw_cfg)
    controller.options = preset.base.motion
    controller._cfg = replace(controller._cfg, close_rate_rad=preset.closing.rate_rad)
    if preset.base.arm_effects:
        raw.backend = GraspServoBackend(raw.backend, raw.config)
    guard = CartesianGuard(raw, controller)
    guard.stable_steps = preset.base.stable_steps
    return CartesianLiftResidual(ContactAudit(guard), controller, preset.limits)

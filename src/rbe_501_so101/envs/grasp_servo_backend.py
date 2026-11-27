"""Experimental manipulation adapter for the existing measured ARM servo model.

Gripper dynamics/delay and object sensing remain uncalibrated. Contact sensing
is privileged simulation truth. No defaults select this adapter automatically.
"""

from dataclasses import replace
from typing import Any

import mujoco
import numpy as np

from rbe_501_so101.config import PickLiftConfig, PRESETS, IKConfig
from rbe_501_so101.envs.calibration import ActuatorCalibration
from rbe_501_so101.envs.perturbed_backend import PerturbedBackend
from rbe_501_so101.envs.servo_effects import ServoEffectsBackend, StictionParams
from rbe_501_so101.kinematics import PositionIKSolver
from rbe_501_so101.kinematics.gravity import GravityTorqueModel


class GraspServoBackend:
    """Preserve object/gripper access while applying arm stiction, clip, delay, sensing.

    Fixed nominal effects use their own RNG so matching placement seeds remain
    matched. Real joint-window constraints are not enabled: this experiment
    evaluates the existing manipulation workspace, not hardware reachability.
    """

    def __init__(self, inner, config: PickLiftConfig, enabled: bool = True) -> None:
        self.inner, self.config = inner, config
        self.arm = self._effects(inner, config) if enabled else inner
        self._state = None
        self._scratch = mujoco.MjData(inner.model)

    @staticmethod
    def _effects(inner, config):
        """Construct fixed nominal ARM effects using existing reaching wrappers."""
        sim = config.sim
        if sim.calibration_path is None:
            raise ValueError("Arm effects require an explicit actuator calibration file")
        params = StictionParams.from_calibration(ActuatorCalibration.load(sim.calibration_path), sim.arm_joints)
        options = PRESETS["residual_ik_sim2real_smooth"].env
        cfg = replace(options.servo, stiction_scale_range=(1.0, 1.0))
        servo = ServoEffectsBackend(inner, params, cfg, GravityTorqueModel(inner.model, inner.ids))
        fk = PositionIKSolver(inner.model, inner.ids, sim, IKConfig())
        sensing = replace(options.perturbation, randomize_delay=False)
        return PerturbedBackend(servo, sensing, sim.control_dt, fk)

    def __getattr__(self, name: str) -> Any:
        """Forward rendering, model/IDs, object placement, and gripper commands."""
        return getattr(self.inner, name)

    def reset(self, rng):
        """Restore arm effects without consuming the object's placement RNG."""
        self._state = self.arm.reset(rng if self.arm is self.inner else np.random.default_rng(0))
        return self._state

    def apply_delta(self, delta):
        """Queue the five arm-command deltas through servo/sensing effects."""
        self.arm.apply_delta(delta)

    def step(self):
        """Advance the existing physical simulation and retain its sensed arm state."""
        self._state = self.arm.step()
        return self._state

    def joint_limits(self):
        """Preserve the arm command bounds of the wrapped backend."""
        return self.arm.joint_limits()

    def apply_gripper_delta(self, delta_rad: float) -> None:
        """Keep the separate moving-jaw command outside the arm delay queue."""
        self.inner.apply_gripper_delta(delta_rad)

    def place_object(self, p_s, yaw_rad: float, mass_kg: float) -> None:
        """Preserve object placement/mass access required by GraspBackend."""
        self.inner.place_object(p_s, yaw_rad, mass_kg)

    def render(self):
        """Forward rendering explicitly for runtime Renderable protocol checks."""
        return self.inner.render()

    def is_open(self) -> bool:
        """Report the actual viewer state rather than treating it as non-renderable."""
        return self.inner.is_open()

    def close(self) -> None:
        """Release all wrapped simulation/rendering resources once."""
        self.arm.close()

    def grasp_state(self):
        """Keep true cube/contact state but compute pinch pose from sensed joints."""
        state = self.inner.grasp_state()
        if self._state is None:
            return state
        self._scratch.qpos[self.inner.ids.qpos_adr] = self._state.qpos
        mujoco.mj_kinematics(self.inner.model, self._scratch)
        site = self.inner.ids.tcp_site
        rotation = self._scratch.site_xmat[site].reshape(3, 3)
        offset = np.asarray(self.config.gripper.pinch_point_b(self.config.object.side_m))
        pinch = self._scratch.site_xpos[site] + rotation @ offset
        return replace(state, robot=self._state, p_pinch=pinch)

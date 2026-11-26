"""Opt-in simulation grasp guard; existing controllers/defaults are unchanged."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from rbe_501_so101.config import ScriptedPickLiftConfig


@dataclass
class GraspGate:
    """Require consecutive two-pad contacts; stop arm motion after contact loss."""

    stable_steps: int = 5
    count: int = 0
    released: bool = False
    aborted: bool = False

    def update(self, grasped: bool, may_release: bool = True) -> bool:
        """Return whether lifting is allowed; an aborted episode cannot rearm."""
        if self.aborted:
            return False
        if self.released and not grasped:
            self.aborted = True
            return False
        self.count = self.count + 1 if grasped else 0
        self.released |= may_release and self.count >= self.stable_steps
        return self.released


class GuardedLift(gym.Wrapper):
    """Filter final executed actions, including learned residuals, at lift time.

    Uses privileged simulation contacts, not a hardware grasp sensor. Waiting
    or aborted states hold the current arm command and close the gripper.
    Existing in-flight delayed commands can still move the arm. This is an
    experiment guard, not a physical safety controller or a grasp recovery plan.
    """

    def __init__(self, env: gym.Env, stable_steps: int = 5) -> None:
        super().__init__(env)
        self.stable_steps = stable_steps
        cfg = ScriptedPickLiftConfig()
        self.lift_start = cfg.approach_steps + cfg.descend_steps + cfg.close_steps
        self.gate = GraspGate(stable_steps)
        self.steps = 0

    def reset(self, **kwargs):
        """Reset contacts/history with the underlying episode."""
        self.gate = GraspGate(self.stable_steps)
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Hold arm commands unless lift has a stable, retained two-pad grasp."""
        action = np.asarray(action, dtype=np.float32).copy()
        raw = self.unwrapped
        allowed = self.gate.update(raw.grasp_state.grasped, may_release=self.steps >= self.lift_start)
        if self.steps >= self.lift_start and not allowed:
            action[:-1] = 0
            error = np.deg2rad(raw.config.gripper.close_deg) - raw.grasp_state.robot.q_cmd[-1]
            action[-1] = np.clip(error / raw.config.task.delta_gripper_max_rad, -1, 1)
        self.steps += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(lift_released=self.gate.released, grasp_abort=self.gate.aborted)
        return obs, reward, terminated, truncated, info

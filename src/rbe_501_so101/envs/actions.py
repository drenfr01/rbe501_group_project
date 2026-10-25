"""Action mappers: normalized policy actions to joint-command changes."""

import numpy as np

from rbe_501_so101.config import ReachTaskConfig
from rbe_501_so101.envs.protocols import ActionMapper
from rbe_501_so101.envs.types import RobotState


def shape_action(action: np.ndarray, exponent: float) -> np.ndarray:
    """`sign(a) * |a|^exponent`: keeps [-1, 1] and the endpoints, refines small actions."""
    action = np.asarray(action, dtype=float)
    return np.sign(action) * np.abs(action) ** exponent


class DeltaActionMapper:
    """Relative command: each step adds the shaped action times `delta_q_max_rad`."""

    def __init__(self, task: ReachTaskConfig) -> None:
        self._delta_max = task.delta_q_max_rad
        self._exponent = task.action_exponent

    def reset(self, joint_goal: np.ndarray | None) -> None:
        """Stateless; the goal is unused."""

    def to_delta(self, action: np.ndarray, state: RobotState) -> np.ndarray:
        """Shaped, scaled relative command."""
        return shape_action(action, self._exponent) * self._delta_max


class ResidualGoalActionMapper:
    """Residual on a joint goal: drive the command toward `goal + residual_max * a`.

    The command moves at most `delta_q_max_rad` per step, so a zero action
    reproduces the IK-hold baseline and the policy only learns the correction.
    """

    def __init__(self, task: ReachTaskConfig) -> None:
        self._delta_max = task.delta_q_max_rad
        self._residual_max = task.residual_max_rad
        self._exponent = task.action_exponent
        self._goal: np.ndarray | None = None

    def reset(self, joint_goal: np.ndarray | None) -> None:
        """Store this episode's joint goal."""
        if joint_goal is None:
            raise ValueError("ResidualGoalActionMapper needs a joint goal")
        self._goal = np.asarray(joint_goal, dtype=float).copy()

    def to_delta(self, action: np.ndarray, state: RobotState) -> np.ndarray:
        """Rate-limited step from the current command toward goal plus residual."""
        if self._goal is None:
            raise RuntimeError("Call reset() before to_delta()")
        desired = self._goal + self._residual_max * shape_action(action, self._exponent)
        return np.clip(desired - state.q_cmd[: len(self._goal)], -self._delta_max, self._delta_max)


def make_action_mapper(task: ReachTaskConfig) -> ActionMapper:
    """Mapper for the configured action mode."""
    match task.action_mode:
        case "delta":
            return DeltaActionMapper(task)
        case "residual_ik":
            return ResidualGoalActionMapper(task)
    raise ValueError(f"Unknown action mode {task.action_mode!r}")

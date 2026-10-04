"""Typed value objects exchanged between environments, backends, and rewards."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Snapshot of the robot after a control step.

    Joint arrays are ordered arm joints first, then the gripper (radians, rad/s).
    `tcp_pos` is the world-frame gripper_site position in meters.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    q_cmd: np.ndarray
    tcp_pos: np.ndarray


@dataclass(frozen=True)
class JointLimits:
    """Lower and upper position limits (radians) for the arm joints."""

    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class RewardTerms:
    """Decomposed reach reward; `total = -distance + bonus - action_penalty`."""

    distance: float
    bonus: float
    action_penalty: float
    total: float

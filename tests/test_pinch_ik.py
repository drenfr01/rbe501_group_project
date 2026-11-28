"""Tests for pinch-point IK: position of p_s = T_sb p_b plus an optional top-down approach axis."""

import numpy as np
import pytest

from rbe_501_so101.config import GripperConfig, ObjectConfig, PinchIKConfig, SimConfig
from rbe_501_so101.envs.grasp_model import build_grasp_model
from rbe_501_so101.envs.model_indices import ModelIndices
from rbe_501_so101.kinematics.pinch_ik import PinchIKSolver, skew

SIM = SimConfig(parked_bodies=())
HOME = np.deg2rad(SIM.initial_deg)
POS_TOL_M = 0.001
TILT_TOL_DEG = 3.0
GRASP_GOAL = np.array([0.02, 0.03, 0.49])
LIFT_GOAL = GRASP_GOAL + np.array([0.0, 0.0, 0.07])


@pytest.fixture(scope="module")
def solver() -> PinchIKSolver:
    """Solver on the pick-and-lift model with the default pinch offset."""
    model = build_grasp_model(SIM, GripperConfig(), ObjectConfig())
    p_b = GripperConfig().pinch_point_b(ObjectConfig().side_m)
    return PinchIKSolver(model, ModelIndices.from_model(model, SIM), PinchIKConfig(), p_b)


def test_skew_matches_cross_product():
    a, b = np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.7, -1.1])
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


def test_pinch_point_is_offset_from_tcp_along_minus_x_b(solver):
    p_tcp, R_sb = solver.tcp_pose(HOME)
    expected = p_tcp + R_sb @ GripperConfig().pinch_point_b(ObjectConfig().side_m)
    np.testing.assert_allclose(solver.forward(HOME), expected)


def test_top_down_solve_reaches_grasp_goal(solver):
    result = solver.solve(GRASP_GOAL, HOME, top_down=True)
    assert result.converged()
    assert np.linalg.norm(solver.forward(result.q) - GRASP_GOAL) < POS_TOL_M
    _, R_sb = solver.tcp_pose(result.q)
    tilt = np.degrees(np.arccos(np.clip(-R_sb[2, 2], -1.0, 1.0)))
    assert tilt < TILT_TOL_DEG


def test_position_only_solve_reaches_lift_goal(solver):
    grasp = solver.solve(GRASP_GOAL, HOME, top_down=True)
    result = solver.solve(LIFT_GOAL, grasp.q, top_down=False)
    assert result.converged()
    assert np.linalg.norm(solver.forward(result.q) - LIFT_GOAL) < POS_TOL_M


def test_solve_leaves_gripper_angle_unchanged(solver):
    q_init = HOME.copy()
    q_init[-1] = 0.3
    assert solver.solve(GRASP_GOAL, q_init, top_down=True).q[-1] == pytest.approx(0.3)

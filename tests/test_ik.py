"""Unit tests for IK helpers that do not require a full solve."""

import numpy as np
import pytest

from rbe_501_so101.config import IKConfig
from rbe_501_so101.kinematics import wrapped_angle_delta
from rbe_501_so101.kinematics.ik import IKResult


def test_wrapped_angle_delta_crosses_pi():
    delta = wrapped_angle_delta(np.pi - 0.1, -np.pi + 0.1)
    assert delta == pytest.approx(0.2)


def test_wrapped_angle_delta_is_zero_for_identical_angles():
    assert wrapped_angle_delta(1.2, 1.2) == pytest.approx(0.0)


def test_ik_result_converged_uses_its_tolerance():
    q = np.zeros(6)
    assert IKResult(q=q, error_m=0.0005, iterations=3, tolerance_m=0.001).converged()
    assert not IKResult(q=q, error_m=0.002, iterations=3, tolerance_m=0.001).converged()


def test_max_step_is_five_degrees():
    assert IKConfig().max_step_rad == pytest.approx(np.deg2rad(5.0))

"""Cartesian acquisition must advance gradually and keep clearance checks explicit."""

import numpy as np
import pytest

from rbe_501_so101.control.cartesian_descent import DescentConfig, next_height, tracking_bias


def test_waypoint_stops_at_goal_and_waits_for_tracking():
    """Do not jump to the final pose or run ahead of measured descent."""
    cfg = DescentConfig()
    assert next_height(.53, .53, .49, cfg) == .528
    assert next_height(.491, .491, .49, cfg) == .49
    assert next_height(.52, .54, .49, cfg) == .52


def test_cartesian_feedback_has_finite_per_axis_budget():
    """Position compensation must remain separate from physical actuator gains."""
    cfg = DescentConfig()
    bias = tracking_bias(np.zeros(3), np.array([1., -1., 1.]), cfg)
    np.testing.assert_array_equal(bias, [-.012, .012, -.012])


def test_invalid_path_sampling_is_rejected():
    """Zero samples must never silently disable geometric checking."""
    with pytest.raises(ValueError):
        DescentConfig(segment_samples=0)


def test_clearance_rejection_keeps_the_previous_target():
    """Rejected kinematic proposals must not overwrite the accepted arm command."""
    from unittest.mock import patch
    from rbe_501_so101.control.cartesian_descent import CartesianDescent
    from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
    from scripts.sim2real.analyze.pick_lift_calibration import make_config

    env = SO101PickLiftEnv(cfg=make_config(.01, True))
    try:
        policy = CartesianDescent.for_env(env)
        obs, _ = env.reset(seed=68000)
        policy(obs)
        target = policy._target.q_arm.copy()
        before = env.backend.data.qpos.copy()
        with patch.object(policy, "_clearance", return_value=(False, -.001)):
            policy._target_for(obs[policy._layout.object_pos] + [0, 0, .05], obs)
        np.testing.assert_array_equal(policy._target.q_arm, target)
        np.testing.assert_array_equal(env.backend.data.qpos, before)
        assert policy.clearance_rejections == 1
    finally:
        env.close()


def test_closing_envelope_blocks_unsafe_measured_posture():
    """An open-jaw-safe pose is insufficient if closing could violate table clearance."""
    from unittest.mock import patch
    from rbe_501_so101.control.cartesian_descent import CartesianDescent
    from rbe_501_so101.rl.lift_residual_limits import PosePrediction
    from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
    from scripts.sim2real.analyze.pick_lift_calibration import make_config

    env = SO101PickLiftEnv(cfg=make_config(.01, True))
    try:
        policy = CartesianDescent.for_env(env)
        obs, _ = env.reset(seed=68000)
        with patch.object(policy.checker, "predict", side_effect=[PosePrediction(.01, .55), PosePrediction(.001, .55)]):
            assert not policy._closure_clear(obs)
        assert policy.closure_rejections == 1
    finally:
        env.close()


def test_late_closure_dwell_holds_even_a_stable_grasp():
    """Five contacts cannot release lift before the configured jaw-settling interval."""
    from dataclasses import replace
    from rbe_501_so101.control.cartesian_descent import CartesianDescent, CartesianGuard
    from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
    from scripts.sim2real.analyze.pick_lift_calibration import make_config

    raw = SO101PickLiftEnv(cfg=make_config(.01, True))
    policy = CartesianDescent.for_env(raw)
    env = CartesianGuard(raw, policy)
    try:
        env.reset(seed=68000)
        raw._state = replace(raw._state, grasped=True)
        before = raw.grasp_state.robot.q_cmd.copy()
        policy.close_start_step, policy._step = 85, 101
        env.steps, env.gate.count = 100, 5
        _, _, _, _, info = env.step(np.ones(6))
        assert not info["lift_released"]
        assert info["closure_settling"]
        np.testing.assert_array_equal(raw.grasp_state.robot.q_cmd[:5], before[:5])
    finally:
        env.close()

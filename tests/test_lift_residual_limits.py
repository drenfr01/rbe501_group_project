"""Experimental residual authority must preserve acquisition and holding clearance."""

import numpy as np

from rbe_501_so101.rl.lift_residual_limits import ResidualLimits, limited_residual, accept_prediction


def test_acquisition_and_gripper_are_protected():
    """Policy corrections cannot disturb approach, descent, closing or the jaw."""
    cfg = ResidualLimits()
    for phase in ("approach", "descend", "close"):
        np.testing.assert_array_equal(limited_residual(np.ones(6), phase, True, cfg), np.zeros(6))
    action = limited_residual(np.ones(6), "lift", True, cfg)
    np.testing.assert_allclose(action[:5], .05)
    assert action[-1] == 0
    np.testing.assert_array_equal(limited_residual(np.ones(6), "lift", False, cfg), np.zeros(6))


def test_clearance_and_holding_reject_unsafe_predictions():
    """A candidate below the clearance margin or holding floor is rejected."""
    cfg = ResidualLimits()
    assert not accept_prediction(clearance=.001, pinch_z=.55, holding_floor=.54, cfg=cfg)
    assert not accept_prediction(clearance=.01, pinch_z=.53, holding_floor=.54, cfg=cfg)
    assert accept_prediction(clearance=.01, pinch_z=.55, holding_floor=.54, cfg=cfg)


def test_zero_residual_preserves_script_and_prediction_does_not_mutate_physics():
    """The opt-in envelope must leave the reference motion exactly reproducible."""
    from rbe_501_so101.rl.limited_lift_training import LimitedLiftPreset, make_env
    from rbe_501_so101.rl.calibrated_pick_lift import make_env as legacy_env

    preset = LimitedLiftPreset()
    env, baseline = make_env(preset), legacy_env(preset.base)
    try:
        obs, _ = env.reset(seed=61010)
        reference, _ = baseline.reset(seed=61010)
        before = env.unwrapped.backend.data.qpos.copy()
        env.checker.predict(env.unwrapped.grasp_state.robot.q_cmd)
        np.testing.assert_array_equal(env.unwrapped.backend.data.qpos, before)
        for _ in range(200):
            obs, *_ = env.step(np.zeros(6))
            reference, *_ = baseline.step(np.zeros(6))
            np.testing.assert_allclose(obs, reference, atol=1e-7)
    finally:
        env.close()
        baseline.close()


def test_v2_repository_round_trip_and_old_policy_still_loads(tmp_path):
    """New serialization must not invalidate the failed v1 experiment."""
    from pathlib import Path
    from rbe_501_so101.config import TrainConfig
    from rbe_501_so101.rl.calibrated_pick_lift import LiftPresetRepository
    from rbe_501_so101.rl.limited_lift_training import LimitedLiftPreset, LimitedLiftRepository

    repo, preset = LimitedLiftRepository(tmp_path / "v2"), LimitedLiftPreset()
    repo.create(preset, TrainConfig(), 0, 10000)
    assert repo.load() == preset
    old = LiftPresetRepository(Path("checkpoints/calibrated_outer_wrist_pilot"))
    if old.path.exists():
        assert old.load().name == "calibrated_outer_wrist_v1"


def test_predicted_height_drop_falls_back_to_script():
    """The holding check must actually filter a proposal, not only report it."""
    from unittest.mock import patch
    from rbe_501_so101.rl.limited_lift_training import LimitedLiftPreset, make_env
    from rbe_501_so101.rl.lift_residual_limits import PosePrediction

    env = make_env(LimitedLiftPreset())
    try:
        env.reset(seed=61010)
        base = env.base_action()
        env._advance_reference()
        env.ever_success = True
        with patch.object(env.checker, "predict", side_effect=[PosePrediction(.01, .56), PosePrediction(.01, .53)]):
            result = env._proposal(base, np.full(6, .05))
        np.testing.assert_array_equal(result, base)
        assert env.counters["holding_rejections"] == 1
    finally:
        env.close()


def test_aborted_grasp_cannot_reenable_residuals():
    """A contact reappearance after a latched abort must not restore policy authority."""
    from rbe_501_so101.rl.limited_lift_training import LimitedLiftPreset, make_env

    env = make_env(LimitedLiftPreset())
    try:
        env.reset(seed=68000)
        for _ in range(101):
            env.step(np.zeros(6))
        assert env.unwrapped.grasp_state.grasped
        env.env.env.gate.released = True
        env.env.env.gate.aborted = True
        env.step(np.ones(6))
        assert env.counters["active_steps"] == 0
    finally:
        env.close()

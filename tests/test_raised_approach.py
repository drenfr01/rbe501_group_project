"""Opt-in acquisition changes must gate closure and preserve the old controller."""

import numpy as np

from rbe_501_so101.control.raised_approach import ApproachConfig, aligned_for_close, corrected_bias


def test_closing_requires_measured_position_and_tilt():
    """A reachable IK target does not authorize closing while measured alignment is poor."""
    cfg = ApproachConfig()
    assert aligned_for_close(np.array([.001, .001, .001]), .01, cfg)
    assert not aligned_for_close(np.array([.012, 0, 0]), .01, cfg)
    assert not aligned_for_close(np.array([0, 0, .012]), .01, cfg)
    assert not aligned_for_close(np.zeros(3), .3, cfg)


def test_alignment_correction_is_bounded_and_opposes_measured_error():
    """Measured Cartesian errors must not create unbounded target offsets."""
    cfg = ApproachConfig(correction_enabled=True)
    bias = corrected_bias(np.zeros(3), np.array([.02, 0, .1]), cfg)
    np.testing.assert_allclose(bias, [-.01, 0, -.015])


def test_disabled_acquisition_options_reproduce_old_actions():
    """The experiment must preserve the previous baseline byte-for-byte in intent."""
    from rbe_501_so101.control.raised_approach import RaisedApproach
    from rbe_501_so101.control.aligned_pick_lift import AlignedPickLift
    from rbe_501_so101.rl.calibrated_pick_lift import outer_wrist_motion
    from scripts.sim2real.analyze.pick_lift_calibration import make_config
    from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv

    env = SO101PickLiftEnv(cfg=make_config(.01, True))
    try:
        baseline = AlignedPickLift.for_env(env, outer_wrist_motion())
        candidate = RaisedApproach.for_env(env, ApproachConfig(raise_first=False, alignment_gate=False))
        obs, _ = env.reset(seed=68006)
        baseline.reset()
        candidate.reset()
        for _ in range(200):
            action = baseline(obs)
            np.testing.assert_allclose(candidate(obs), action, atol=1e-7)
            obs, *_ = env.step(action)
    finally:
        env.close()


def test_acquisition_abort_keeps_jaw_open_despite_lift_guard():
    """An alignment rejection must not be overridden by the old guard's closing command."""
    from rbe_501_so101.control.raised_approach import RaisedApproach, AcquisitionGuard
    from scripts.sim2real.analyze.pick_lift_calibration import make_config
    from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv

    raw = SO101PickLiftEnv(cfg=make_config(.01, True))
    controller = RaisedApproach.for_env(raw)
    env = AcquisitionGuard(raw, controller)
    try:
        env.reset(seed=68006)
        env.step(np.array([0., 0., 0., 0., 0., -1.]))
        before = raw.grasp_state.robot.q_cmd.copy()
        controller.aborted = True
        env.steps = 100
        _, _, _, _, info = env.step(np.ones(6))
        after = raw.grasp_state.robot.q_cmd
        np.testing.assert_array_equal(after[:5], before[:5])
        assert after[-1] > before[-1]
        assert info["acquisition_abort"]
    finally:
        env.close()

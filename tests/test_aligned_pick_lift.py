"""Control-side alignment/feedforward experiments must retain the old baseline."""

import numpy as np

from rbe_501_so101.control.aligned_pick_lift import AlignedPickLift, MotionOptions, quarter_turn_error, compensate
from rbe_501_so101.control import ScriptedPickLift
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv


def test_cube_yaw_error_uses_quarter_turn_symmetry():
    """Equivalent square faces should not request a large wrist rotation."""
    assert abs(quarter_turn_error(np.pi, 0)) < 1e-12
    assert abs(quarter_turn_error(np.pi / 2, 0)) < 1e-12
    assert abs(quarter_turn_error(.2, 0) - .2) < 1e-12


def test_feedforward_is_bounded_and_respects_joint_limits():
    """Gravity correction changes commanded angles, never actuator gains."""
    q = np.zeros(5)
    limits = np.tile([-.05, .05], (5, 1))
    result = compensate(q, np.ones(5), np.ones(5), limits, .1)
    np.testing.assert_array_equal(result, np.full(5, .05))


def test_disabled_options_reproduce_scripted_actions():
    """New controls are opt-in; no changes to the old sequence are required."""
    env = SO101PickLiftEnv()
    baseline = ScriptedPickLift.for_env(env)
    candidate = AlignedPickLift.for_env(env, MotionOptions("disabled"))
    try:
        obs, _ = env.reset(seed=60000)
        baseline.reset()
        candidate.reset()
        for _ in range(200):
            action = baseline(obs)
            np.testing.assert_allclose(candidate(obs), action, atol=1e-7)
            obs, *_ = env.step(action)
    finally:
        env.close()


def test_radial_yaw_seed_aligns_with_the_current_cube_placement_rule():
    """An oblique spawn should not retain the original nearly-world-fixed jaw yaw."""
    env = SO101PickLiftEnv()
    try:
        obs, _ = env.reset(seed=60002)
        policy = AlignedPickLift.for_env(env, MotionOptions("aligned", radial_jaw_alignment=True))
        policy(obs)
        q = env.grasp_state.robot.q_cmd.copy()
        q[:5] = policy._target.q_arm
        _, rotation = policy._solver.tcp_pose(q)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        offset = obs[env.layout.object_pos][:2] - policy._solver._pan_axis_xy
        cube_yaw = np.arctan2(offset[1], offset[0])
        assert abs(quarter_turn_error(cube_yaw, yaw)) < np.deg2rad(5)
    finally:
        env.close()

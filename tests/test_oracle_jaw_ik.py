"""The experimental jaw objective must use a correct rotational Jacobian."""

import numpy as np

from rbe_501_so101.kinematics.oracle_jaw_ik import JawIKConfig, JawIKSolver, nearest_face_axis


def test_square_symmetry_uses_nearest_face_axis() -> None:
    """Choose equivalent cube faces without unnecessarily reversing the wrist."""
    axis = nearest_face_axis(np.deg2rad(80), 0.)
    np.testing.assert_allclose(axis, [np.cos(np.deg2rad(-10)), np.sin(np.deg2rad(-10)), 0], atol=1e-12)


def test_axis_jacobian_matches_finite_difference() -> None:
    """Differentiate the TCP X axis using MuJoCo's angular Jacobian."""
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset, make_env

    env = make_env(CartesianZeroPreset())
    try:
        env.reset(seed=82000)
        base = env._controller._solver
        solver = JawIKSolver(base._model, base._ids, base._cfg, base._p_b,
                             lambda: .4, JawIKConfig())
        q = env.unwrapped.grasp_state.robot.qpos.copy()
        _, rotation = solver.tcp_pose(q)
        analytic = solver.axis_jacobian(rotation[:, 0])
        numeric = np.zeros_like(analytic)
        for joint in range(5):
            perturbed = q.copy()
            perturbed[joint] += 1e-6
            _, changed = solver.tcp_pose(perturbed)
            numeric[:, joint] = (changed[:, 0] - rotation[:, 0]) / 1e-6
        np.testing.assert_allclose(analytic, numeric, atol=1e-6)
    finally:
        env.close()


def test_oracle_reaches_recorded_rotated_cube_without_reversing_jaw() -> None:
    """Respect square symmetry and keep gripper command independent of the arm solve."""
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset, make_env

    env = make_env(CartesianZeroPreset())
    try:
        env.reset(seed=82000)
        base, raw = env._controller._solver, env.unwrapped
        center = raw.grasp_state.obj.p_s.copy()
        bearing = float(np.arctan2(*(center[:2] - raw.backend.pan_axis_xy)[::-1]))
        solver = JawIKSolver(base._model, base._ids, base._cfg, base._p_b,
                             lambda: bearing + np.pi / 4, JawIKConfig())
        initial = raw.grasp_state.robot.qpos.copy()
        result = solver.solve(center, initial, True)
        assert result.converged()
        assert result.q[-1] == initial[-1]
        assert result.axis_error_rad < np.deg2rad(1)
        solver.reset_axis()
        assert solver.desired_axis is None and solver.last_result is None
    finally:
        env.close()


def test_oracle_controller_preserves_model_and_acquisition_guard() -> None:
    """A privileged planning input must not alter the compiled physics or protections."""
    from rbe_501_so101.control.oracle_acquisition import make_oracle_env
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset, make_env

    baseline, oracle = make_env(CartesianZeroPreset()), make_oracle_env(CartesianZeroPreset())
    try:
        a, b = baseline.unwrapped.backend.model, oracle.unwrapped.backend.model
        for name in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange",
                     "jnt_range", "body_mass", "body_inertia", "body_gravcomp", "geom_friction",
                     "geom_contype", "geom_conaffinity"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert oracle._controller.descent == baseline._controller.descent
        assert oracle._controller.acquisition == baseline._controller.acquisition
        assert oracle.unwrapped.config == baseline.unwrapped.config
    finally:
        baseline.close()
        oracle.close()

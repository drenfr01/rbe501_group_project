"""Unit tests for MujocoSo101Backend: protocol, delta integration, stepping, randomization."""

import numpy as np

from rbe_501_so101.config import DynamicsRandomizationConfig, ReachTaskConfig, SimConfig
from rbe_501_so101.envs.protocols import RobotBackend
from rbe_501_so101.envs.types import RobotState

N_ARM = len(SimConfig().arm_joints)
N_JOINTS = len(SimConfig().joints)
DELTA_MAX = ReachTaskConfig().delta_q_max_rad
OVERSIZED_DELTA_RAD = 100.0


def test_backend_implements_robot_backend_protocol(backend):
    assert isinstance(backend, RobotBackend)


def test_reset_commands_home_pose(backend, rng, sim_cfg):
    state = backend.reset(rng)
    np.testing.assert_allclose(state.q_cmd, np.deg2rad(sim_cfg.initial_deg))
    np.testing.assert_allclose(state.qpos, np.deg2rad(sim_cfg.initial_deg))


def test_zero_delta_leaves_command_unchanged(backend, rng):
    before = backend.reset(rng).q_cmd.copy()
    backend.apply_delta(np.zeros(N_ARM))
    np.testing.assert_array_equal(backend.step().q_cmd, before)


def test_full_delta_changes_command_by_delta_max(backend, rng):
    before = backend.reset(rng).q_cmd.copy()
    backend.apply_delta(np.full(N_ARM, DELTA_MAX))
    after = backend.step().q_cmd
    np.testing.assert_allclose(after[:N_ARM] - before[:N_ARM], DELTA_MAX)
    assert after[N_ARM] == before[N_ARM]


def test_command_is_clipped_to_joint_limits(backend, rng):
    backend.reset(rng)
    limits = backend.joint_limits()
    backend.apply_delta(np.full(N_ARM, OVERSIZED_DELTA_RAD))
    np.testing.assert_allclose(backend.step().q_cmd[:N_ARM], limits.upper)
    backend.apply_delta(np.full(N_ARM, -2 * OVERSIZED_DELTA_RAD))
    np.testing.assert_allclose(backend.step().q_cmd[:N_ARM], limits.lower)


def test_joint_limits_match_model(backend, raw_model, sim_cfg):
    limits = backend.joint_limits()
    expected = np.array([raw_model.joint(name).range for name in sim_cfg.arm_joints])
    np.testing.assert_allclose(limits.lower, expected[:, 0])
    np.testing.assert_allclose(limits.upper, expected[:, 1])


def test_step_returns_finite_state_with_expected_shapes(backend, rng):
    backend.reset(rng)
    state = backend.step()
    assert isinstance(state, RobotState)
    assert state.qpos.shape == state.qvel.shape == state.q_cmd.shape == (N_JOINTS,)
    assert state.tcp_pos.shape == (3,)
    assert all(np.all(np.isfinite(a)) for a in (state.qpos, state.qvel, state.tcp_pos))


def test_step_advances_one_control_period(backend, rng, sim_cfg):
    backend.reset(rng)
    backend.step()
    expected = sim_cfg.n_substeps * backend.model.opt.timestep
    np.testing.assert_allclose(backend.data.time, expected)


def test_reset_without_randomization_keeps_nominal_dynamics(backend, rng, raw_model):
    backend.reset(rng)
    np.testing.assert_allclose(backend.model.actuator_gainprm[:, 0], raw_model.actuator_gainprm[:, 0])
    np.testing.assert_allclose(backend.model.actuator_biasprm[:, :3], raw_model.actuator_biasprm[:, :3])


def test_randomization_scales_gains_within_configured_range(sim_cfg, rng, raw_model):
    from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend

    cfg = DynamicsRandomizationConfig(enabled=True)
    backend = MujocoSo101Backend(sim_cfg, randomization=cfg)
    backend.reset(rng)
    kp_scale = backend.model.actuator_gainprm[:, 0] / raw_model.actuator_gainprm[:, 0]
    kv_scale = backend.model.actuator_biasprm[:, 2] / raw_model.actuator_biasprm[:, 2]
    assert np.all((kp_scale >= cfg.kp_scale_range[0]) & (kp_scale <= cfg.kp_scale_range[1]))
    assert np.all((kv_scale >= cfg.damping_scale_range[0]) & (kv_scale <= cfg.damping_scale_range[1]))
    assert not np.allclose(kp_scale, 1.0)
    np.testing.assert_allclose(backend.model.actuator_biasprm[:, 1], -backend.model.actuator_gainprm[:, 0])
    backend.close()


def test_randomization_adds_bounded_payload(sim_cfg, rng, raw_model):
    from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend

    cfg = DynamicsRandomizationConfig(enabled=True)
    backend = MujocoSo101Backend(sim_cfg, randomization=cfg)
    backend.reset(rng)
    body = raw_model.body(sim_cfg.payload_body).id
    added = backend.model.body_mass[body] - raw_model.body_mass[body]
    assert cfg.payload_mass_range_kg[0] <= added <= cfg.payload_mass_range_kg[1]
    backend.close()


def test_randomization_does_not_compound_across_resets(sim_cfg, raw_model):
    from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend

    cfg = DynamicsRandomizationConfig(enabled=True)
    backend = MujocoSo101Backend(sim_cfg, randomization=cfg)
    gen = np.random.default_rng(0)
    for _ in range(20):
        backend.reset(gen)
    kp_scale = backend.model.actuator_gainprm[:, 0] / raw_model.actuator_gainprm[:, 0]
    assert np.all((kp_scale >= cfg.kp_scale_range[0]) & (kp_scale <= cfg.kp_scale_range[1]))
    backend.close()

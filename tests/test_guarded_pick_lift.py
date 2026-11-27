"""Opt-in grasp guards and arm effects must preserve baseline task semantics."""

import numpy as np
import pytest
from dataclasses import replace

from rbe_501_so101.config import DEFAULT_CALIBRATION_FILE, PickLiftConfig, resolve_project_path

from rbe_501_so101.control.guarded_pick_lift import GraspGate, GuardedLift
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from rbe_501_so101.envs.protocols import GraspBackend, Renderable


def measured_config():
    """The measured arm JSON, explicitly opted into the manipulation config."""
    config = PickLiftConfig()
    return replace(config, sim=replace(config.sim, calibration_path=resolve_project_path(DEFAULT_CALIBRATION_FILE)))


def test_adapter_preserves_runtime_grasp_and_render_protocols():
    """Python 3.12 protocol checks need explicit methods, not just __getattr__."""
    env = SO101PickLiftEnv(cfg=measured_config())
    try:
        adapter = GraspServoBackend(env.backend, env.config)
        assert isinstance(adapter, GraspBackend)
        assert isinstance(adapter, Renderable)
    finally:
        env.close()


def test_arm_effects_require_an_explicit_calibration():
    """A missing fit must not silently create a supposedly calibrated arm."""
    env = SO101PickLiftEnv()
    try:
        with pytest.raises(ValueError, match="calibration"):
            GraspServoBackend(env.backend, env.config)
    finally:
        env.close()


def test_guard_blocks_final_arm_action_without_a_grasp():
    """The guard filters actual commands, not just the scripted base action."""
    raw = SO101PickLiftEnv()
    env = GuardedLift(raw)
    try:
        env.reset(seed=0)
        before = raw.grasp_state.robot.q_cmd.copy()
        env.steps = env.lift_start
        env.step(np.ones(6, dtype=np.float32))
        after = raw.grasp_state.robot.q_cmd
        np.testing.assert_array_equal(after[:5], before[:5])
        assert after[-1] < before[-1]
    finally:
        env.close()


def test_gate_needs_consecutive_contacts_and_latches_loss():
    """One contact blip cannot release lift, and a lost grasp aborts arm motion."""
    gate = GraspGate(stable_steps=2)
    assert not gate.update(True)
    assert not gate.update(False)
    assert not gate.update(True)
    assert gate.update(True)
    assert not gate.update(False)
    assert gate.aborted
    assert not gate.update(True)


def test_contact_changes_during_closing_do_not_abort_before_lift():
    """The jaw may settle during closing; only an actual lift can latch a loss."""
    gate = GraspGate(stable_steps=2)
    assert not gate.update(True, may_release=False)
    assert not gate.update(True, may_release=False)
    assert not gate.released
    assert not gate.update(False, may_release=False)
    assert not gate.aborted


def test_disabled_arm_effects_preserve_gripper_cube_and_observations():
    """Wrapping the arm must not hide grasp state or change placement RNG."""
    baseline, wrapped = SO101PickLiftEnv(), SO101PickLiftEnv()
    wrapped.backend = GraspServoBackend(wrapped.backend, wrapped.config, enabled=False)
    try:
        original, _ = baseline.reset(seed=20002)
        adapted, _ = wrapped.reset(seed=20002)
        np.testing.assert_allclose(adapted, original, atol=1e-7)
        for _ in range(10):
            action = np.array([0.01, 0, 0, 0, 0, -0.1], dtype=np.float32)
            original, *_ = baseline.step(action)
            adapted, *_ = wrapped.step(action)
            np.testing.assert_allclose(adapted, original, atol=1e-7)
        assert wrapped.grasp_state.obj.mass_kg == baseline.grasp_state.obj.mass_kg
    finally:
        baseline.close()
        wrapped.close()


def test_fixed_arm_effects_do_not_change_cube_placement_seed():
    """Effects may change sensed joints, but must not resample the matched object."""
    baseline, wrapped = SO101PickLiftEnv(), SO101PickLiftEnv(cfg=measured_config())
    wrapped.backend = GraspServoBackend(wrapped.backend, wrapped.config, enabled=True)
    try:
        baseline.reset(seed=20002)
        wrapped.reset(seed=20002)
        np.testing.assert_array_equal(baseline.grasp_state.obj.p_s, wrapped.grasp_state.obj.p_s)
        assert wrapped.grasp_state.robot.qpos.shape == (6,)
    finally:
        baseline.close()
        wrapped.close()


def test_measured_arm_effects_report_delayed_commands_and_quantized_joints():
    """One-step arm delay must not delay or erase the separate gripper command."""
    from dataclasses import replace
    from rbe_501_so101.config import DEFAULT_CALIBRATION_FILE, ENCODER_QUANTIZATION_RAD, PickLiftConfig, resolve_project_path

    config = PickLiftConfig()
    config = replace(config, sim=replace(config.sim, calibration_path=resolve_project_path(DEFAULT_CALIBRATION_FILE)))
    env = SO101PickLiftEnv(cfg=config)
    env.backend = GraspServoBackend(env.backend, config)
    try:
        env.reset(seed=20002)
        before = env.grasp_state.robot.q_cmd.copy()
        env.step(np.array([0.1, 0, 0, 0, 0, -0.1], dtype=np.float32))
        state = env.grasp_state.robot
        assert state.q_cmd[0] > before[0]
        assert state.q_cmd[-1] < before[-1]
        inner_command = env.backend.inner.command
        assert inner_command[0] == before[0]
        np.testing.assert_allclose(state.qpos / ENCODER_QUANTIZATION_RAD,
                                   np.round(state.qpos / ENCODER_QUANTIZATION_RAD), atol=1e-10)
    finally:
        env.close()

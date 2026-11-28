"""Tests for the joint-torque analysis: equation-of-motion split, payload model, sag, capacity, feedforward."""

import numpy as np
import pytest

from rbe_501_so101.config import ObjectMassConfig
from rbe_501_so101.pick_place.env import PlaceRun, ScriptedPickPlace, SO101PickPlaceEnv, make_env
from rbe_501_so101.pick_place.torque import (
    LIFT,
    ELBOW,
    WRIST_FLEX,
    ArmStatics,
    FeedforwardPickPlace,
    Trace,
    capacity_map,
    record_episode,
)

SEED = 20_000
LOADED = (LIFT, ELBOW, WRIST_FLEX)
"""Pitch joints that carry the weight; pan and roll axes are vertical / along the wrist and see almost none."""


def _env(mass_kg: float) -> SO101PickPlaceEnv:
    """Bare pick-and-place env with a fixed cube mass."""
    return SO101PickPlaceEnv(mass_cfg=ObjectMassConfig(mode="fixed", fixed_kg=mass_kg))


def _trace(mass_kg: float, controller_type: type[ScriptedPickPlace] = ScriptedPickPlace) -> Trace:
    """One recorded episode of a scripted controller at a mass."""
    env = _env(mass_kg)
    controller = controller_type.for_env(env)
    trace = record_episode(env, controller, controller, SEED, controller_type.__name__)
    env.close()
    return trace


@pytest.fixture(scope="module")
def half_kg() -> Trace:
    """Scripted episode with a 0.5 kg cube (held the whole carry, no saturation)."""
    return _trace(0.5)


@pytest.fixture(scope="module")
def one_kg() -> Trace:
    """Scripted episode with a 1.0 kg cube."""
    return _trace(1.0)


def test_terms_add_up_to_the_actuator_torque(half_kg):
    assert np.abs(half_kg.terms.residual).max() < 1e-3  # contact-solver tolerance, vs ~1 N·m loads


def test_arm_at_rest_is_held_by_gravity_torque_and_sags_by_it_over_kp():
    env = _env(0.05)
    env.reset(seed=SEED)
    statics = ArmStatics(env)
    for _ in range(60):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    s = env.backend.grasp_state().robot
    g, _ = statics.loads(s.qpos)
    actuator = env.backend.data.qfrc_actuator[statics.dof]
    np.testing.assert_allclose(actuator, g, atol=2e-3)
    np.testing.assert_allclose((s.q_cmd - s.qpos)[:5], g / statics.kp, atol=2e-4)
    env.close()


def test_hold_window_is_the_end_of_carry_with_the_cube_grasped(half_kg):
    hold = half_kg.hold
    assert hold.sum() > 0
    assert set(half_kg.phase[hold]) == {"carry"}
    assert half_kg.grasped[hold].all()


def test_payload_model_predicts_the_contact_torque_while_holding(half_kg):
    hold = half_kg.hold
    measured = half_kg.terms.contact[hold][:, LOADED].mean(axis=0)
    model = half_kg.payload_model[hold][:, LOADED].mean(axis=0)
    np.testing.assert_allclose(model, measured, rtol=0.10, atol=0.02)


def test_predicted_sag_matches_the_measured_sag_while_holding(half_kg):
    summary = half_kg.summary()
    np.testing.assert_allclose(summary.sag_predicted[[LIFT, ELBOW]], summary.sag_measured[[LIFT, ELBOW]],
                               rtol=0.15, atol=2e-3)


def test_heavier_cube_needs_more_shoulder_torque_and_drops_further(half_kg, one_kg):
    light, heavy = half_kg.summary(), one_kg.summary()
    assert heavy.peak_nm[LIFT] > light.peak_nm[LIFT]
    assert heavy.drop_measured_m > light.drop_measured_m


def test_capacity_is_the_mass_where_the_first_joint_hits_its_limit():
    env = _env(0.05)
    env.reset(seed=SEED)
    statics = ArmStatics(env)
    q = env.backend.grasp_state().robot.qpos
    limit = 2.0
    m_max, joint = statics.capacity(q, limit)
    g, b = statics.loads(q)
    torque = np.abs(g + m_max * b)
    assert torque[joint] == pytest.approx(limit)
    assert (torque <= limit + 1e-9).all()
    env.close()


def test_capacity_shrinks_with_reach():
    env = _env(0.05)
    env.reset(seed=SEED)
    cmap = capacity_map(env, radii=(0.12, 0.24), heights=(0.07,), limit_nm=2.9)
    assert cmap.m_max_kg[0, 0] > cmap.m_max_kg[0, 1] > 0
    env.close()


def test_feedforward_cancels_most_of_the_carry_drop(one_kg):
    compensated = _trace(1.0, FeedforwardPickPlace).summary()
    assert abs(compensated.drop_measured_m) < 0.3 * one_kg.summary().drop_measured_m


def test_drop_splits_into_arm_sag_and_slip_in_the_grip(one_kg):
    s = one_kg.summary()
    assert s.drop_arm_m + s.slip_m == pytest.approx(s.drop_measured_m)
    assert s.drop_arm_m == pytest.approx(s.drop_predicted_m, abs=3e-3)


def test_grip_friction_holds_the_cube_with_margin(one_kg):
    assert one_kg.summary().grip_margin > 1.0


def test_place_window_is_the_end_of_lowering_with_the_cube_grasped(half_kg):
    place = half_kg.place
    assert place.sum() > 0
    assert set(half_kg.phase[place]) == {"lower"}
    assert half_kg.grasped[place].all()


def test_sag_pushes_the_cube_sideways_more_for_a_heavier_cube(half_kg, one_kg):
    assert one_kg.summary().place_shift_predicted_m > half_kg.summary().place_shift_predicted_m > 0


def test_scripted_run_has_no_residual(half_kg):
    assert not half_kg.residual.any()


def test_residual_can_offset_the_command_by_at_most_scale_times_delta_max():
    run = PlaceRun(ObjectMassConfig(mode="fixed", fixed_kg=0.5))
    env = make_env(run.mass, run)
    full_push = lambda obs: np.ones(env.action_space.shape, dtype=np.float32)  # noqa: E731
    trace = record_episode(env, full_push, env._controller, SEED, "push")
    cap = run.residual_scale * env.unwrapped.config.task.delta_q_max_rad
    np.testing.assert_allclose(trace.residual[trace.place][:, LOADED], cap, atol=1e-6)
    offset = (trace.q_cmd - trace.q_target)[trace.place][:, [LIFT, ELBOW]]
    assert np.abs(offset).max() <= cap + 1e-6
    env.close()

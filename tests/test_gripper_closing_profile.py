"""Closing-profile experiments must limit speed and latch contact deterministically."""

import numpy as np

from rbe_501_so101.control.gripper_closing import ClosingProfile, ClosingLatch, closing_action


def test_closing_speed_is_limited_without_changing_actuator_gains():
    """A slower profile changes only the commanded angle increment."""
    profile = ClosingProfile("slow", rate_rad=.025)
    assert closing_action(np.deg2rad(45), np.deg2rad(-10), profile, .1) == -.25


def test_contact_latch_waits_for_consecutive_contacts_and_holds_command():
    """A transient first contact must not silently become a stable-grasp latch."""
    profile = ClosingProfile("latch", stop_on_contact=True, stable_steps=2)
    latch = ClosingLatch(profile)
    assert latch.target(.2, True) == np.deg2rad(-10)
    latch.target(.15, False)
    assert latch.target(.1, True) == np.deg2rad(-10)
    assert latch.target(.05, True) == .05
    assert latch.target(0., False) == .05


def test_normalized_closing_action_never_exceeds_environment_bounds():
    """Even a rate larger than one full action must respect [-1, 1]."""
    assert closing_action(1., -1., ClosingProfile("fast", rate_rad=.3), .1) == -1.


def test_isolated_closing_keeps_arm_commands_fixed():
    """A closing comparison must not secretly continue tracking the arm target."""
    from scripts.sim2real.analyze.gripper_closing_check import build, prepare, close_stage, close_metrics, ClosingProtocol
    from rbe_501_so101.envs.model_indices import descendant_body_ids

    env, policy = build(.005, True)
    try:
        cfg = ClosingProtocol()
        prepared = prepare(env, policy, 68006, cfg)
        assert prepared is not None
        raw = env.unwrapped
        bodies = descendant_body_ids(raw.backend.model, raw.config.sim.robot_root_body)
        _, _, frames, _ = close_stage(raw, ClosingProfile("zero", target_deg=0.), prepared, cfg, bodies)
        metrics = close_metrics(frames, cfg)
        assert metrics["max_arm_command_drift_rad"] == 0
        assert metrics["stable_end"]
    finally:
        env.close()


def test_new_endpoint_does_not_change_compiled_physical_model():
    """A controller endpoint must not become a hidden gain, joint-limit or collision edit."""
    from scripts.sim2real.analyze.gripper_closing_check import build_end_to_end

    reference, _ = build_end_to_end(ClosingProfile("minus10"), .01, True)
    candidate, _ = build_end_to_end(ClosingProfile("zero", target_deg=0.), .01, True)
    try:
        old, new = reference.unwrapped.backend.model, candidate.unwrapped.backend.model
        for name in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange",
                     "jnt_range", "body_mass", "body_inertia", "body_gravcomp", "geom_friction",
                     "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity"):
            np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
        assert candidate.unwrapped.config.task.max_episode_steps == 200
    finally:
        reference.close()
        candidate.close()

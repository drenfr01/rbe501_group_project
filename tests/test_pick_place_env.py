"""Full-task environment must preserve the script and distinguish placement from lift."""

from dataclasses import replace

import numpy as np
import pytest

from rbe_501_so101.envs.pick_place_env import SO101PickPlaceEnv
from rbe_501_so101.envs.pick_place_config import PickPlaceConfig, PlacementStatus, placement_success


def test_lift_or_target_proximity_alone_is_not_complete_success() -> None:
    """Require release, support, retreat and the final observation window."""
    cfg = PickPlaceConfig()
    status = PlacementStatus(True, True, True, True, True, False, .001, .001, .001)
    assert placement_success(status, cfg)
    for field in ("lifted", "released", "retreated", "settled", "supported"):
        assert not placement_success(replace(status, **{field: False}), cfg)
    assert not placement_success(replace(status, grasped=True), cfg)
    assert not placement_success(replace(status, angular_speed_rad_s=1.), cfg)


def test_reset_is_seeded_and_observes_destination_and_orientation() -> None:
    """New observations must expose the full task rather than reusing the old 33-vector."""
    env = SO101PickPlaceEnv()
    try:
        a, _ = env.reset(seed=92000, options={"yaw_offset_deg": 30.})
        b, _ = env.reset(seed=92000, options={"yaw_offset_deg": 30.})
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a[env.layout.get("destination")], env.sequence.target)
        assert len(a[env.layout.get("cube_quaternion")]) == 4
        assert env.observation_space.contains(a)
        assert env.action_space.shape == (3,)
    finally:
        env.close()


def test_horizon_truncates_and_requires_reset() -> None:
    """A time limit is truncation, not successful task termination."""
    env = SO101PickPlaceEnv(replace(PickPlaceConfig(), max_episode_steps=3))
    try:
        env.reset(seed=92000)
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(np.zeros(3))
        assert truncated and not terminated and not info["is_success"]
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(3))
    finally:
        env.close()


def test_zero_residual_matches_validated_complete_sequence() -> None:
    """A new Gym state machine must reproduce every physical pose of the validated script."""
    from rbe_501_so101.control.contact_tracked_place import ContactTrackedCarryPlace
    from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
    from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset
    from scripts.sim2real.analyze.pick_lift_coverage import DiagnosticPlacer, PoseCase

    preset = CartesianZeroPreset()
    preset = replace(preset, base=replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=.005)))
    env = SO101PickPlaceEnv(replace(PickPlaceConfig(), preset=preset))
    baseline = make_timed_env(preset)
    raw = baseline.unwrapped
    raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z,
                                  PoseCase("recorded", yaw_offset_deg=30))
    script = ContactTrackedCarryPlace(baseline)
    try:
        assert script.acquire(89000) and script.run_transfer()
        env.reset(seed=89000, options={"yaw_offset_deg": 30.})
        for expected in script.frames:
            obs, reward, terminated, truncated, info = env.step(np.zeros(3))
            np.testing.assert_array_equal(env.sequence.backend.data.qpos, expected.qpos)
            assert np.isfinite(reward) and env.observation_space.contains(obs)
        assert terminated and not truncated and info["is_success"]
        assert info["phase"] == "settle" and not info["grasped"]
    finally:
        env.close()
        baseline.close()


def test_policy_authority_is_masked_until_transfer_and_bounded() -> None:
    """Random acquisition actions must not bypass the validated grasp sequence."""
    env = SO101PickPlaceEnv()
    try:
        env.reset(seed=92000)
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(np.ones(3))
            assert not info["residual_applied"]
        assert not terminated and not truncated and info["lifted"]
        _, _, _, _, info = env.step(np.ones(3))
        assert info["residual_applied"]
        assert 0 < info["residual_norm_m"] <= env.config.residual_scale_m
        assert not info["is_success"]
    finally:
        env.close()


def test_invalid_goal_or_action_cannot_leave_a_steppable_invalid_episode() -> None:
    """Validate explicit destinations and reject nonfinite actions before physical advancement."""
    env = SO101PickPlaceEnv()
    try:
        with pytest.raises(ValueError, match="Destination"):
            env.reset(seed=92000, options={"destination_xy": [9., 9.]})
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(3))
        env.reset(seed=92000)
        before = env.sequence.backend.data.time
        with pytest.raises(ValueError, match="finite"):
            env.step([float("nan"), 0., 0.])
        assert env.sequence.backend.data.time == before
    finally:
        env.close()


def test_rejected_residual_falls_back_without_an_extra_physics_step() -> None:
    """A rejected proposal must restore controller state before executing the script once."""
    env = SO101PickPlaceEnv()
    try:
        env.reset(seed=92000)
        for _ in range(200):
            env.step(np.zeros(3))
        original = env.sequence.move
        calls = []

        def reject_once(goal, phase):
            """Simulate a pre-execution clearance rejection, then use the real safe fallback."""
            calls.append(goal.copy())
            if len(calls) == 1:
                env.sequence.reference = np.ones(3)
                env.sequence.tracking_bias[:] = .006
                env.sequence.failure = "transfer_clearance_rejected"
                return False
            return original(goal, phase)

        env.sequence.move = reject_once
        before = env.sequence.backend.data.time
        _, _, terminated, _, info = env.step(np.ones(3))
        assert len(calls) == 2 and not terminated
        assert env.sequence.backend.data.time - before == pytest.approx(env.sequence.raw.control_dt)
        assert info["residual_rejections"] == 1 and not info["residual_applied"]
    finally:
        env.close()

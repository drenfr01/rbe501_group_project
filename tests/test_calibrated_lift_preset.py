"""The calibrated pilot must reconstruct its own controller and reject drift."""

from dataclasses import replace

import numpy as np
import pytest

from rbe_501_so101.rl.calibrated_pick_lift import CalibratedLiftPreset, LiftPresetRepository, make_env
from rbe_501_so101.config import TrainConfig


def test_preset_round_trip_and_fresh_directory(tmp_path):
    """Save all physics/controller choices and prevent accidental checkpoint reuse."""
    repo = LiftPresetRepository(tmp_path / "run")
    preset = CalibratedLiftPreset()
    repo.create(preset, TrainConfig(), seed=0, steps=10000)
    assert repo.load() == preset
    with pytest.raises(FileExistsError):
        repo.create(preset, TrainConfig(), seed=0, steps=10000)


def test_loading_rejects_changed_configuration(tmp_path):
    """A preset file cannot silently use different runtime physics defaults."""
    repo = LiftPresetRepository(tmp_path / "run")
    repo.create(CalibratedLiftPreset(), TrainConfig(), seed=0, steps=10000)
    text = repo.path.read_text().replace('"stable_steps": 5', '"stable_steps": 6', 1)
    repo.path.write_text(text)
    with pytest.raises(ValueError, match="snapshot"):
        repo.load()


def test_zero_residual_matches_validated_scripted_setup():
    """The training stack must preserve the evaluated controller and wrapper order."""
    from scripts.sim2real.analyze.pick_lift_motion_ablation import build, motion_options

    preset = CalibratedLiftPreset()
    mass = replace(preset.mass, mode="fixed", fixed_kg=.01)
    env = make_env(replace(preset, mass=mass))
    reference, policy, _ = build(motion_options("outer_wrist"), True, .01, "scripted")
    try:
        obs, _ = env.reset(seed=61010)
        expected, _ = reference.reset(seed=61010)
        policy.reset()
        for _ in range(200):
            obs, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
            expected, *_ = reference.step(policy(expected))
            np.testing.assert_allclose(obs, expected, atol=1e-7)
        assert info["is_success"]
        assert info["robot_table_contact_steps"] == 0
    finally:
        env.close()
        reference.close()

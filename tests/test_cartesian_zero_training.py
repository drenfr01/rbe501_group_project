"""The new RL preset must reproduce the validated 0-degree scripted controller."""

from dataclasses import replace

import numpy as np
import pytest

from rbe_501_so101.config import TrainConfig
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset, CartesianZeroRepository, make_env


def test_preset_round_trip_and_new_directory_only(tmp_path):
    """Do not resume older or differently configured policies accidentally."""
    repo, preset = CartesianZeroRepository(tmp_path / "v3"), CartesianZeroPreset()
    repo.create(preset, TrainConfig(), 0, 10000)
    assert repo.load() == preset
    with pytest.raises(FileExistsError):
        repo.create(preset, TrainConfig(), 0, 10000)
    text = repo.path.read_text().replace('"target_deg": 0.0', '"target_deg": -10.0', 1)
    repo.path.write_text(text)
    with pytest.raises(ValueError, match="snapshot"):
        repo.load()


@pytest.mark.parametrize("mass", [.005, .01])
def test_zero_residual_matches_validated_sequence(mass):
    """Every observation must match the original 200-step scripted replay."""
    from rbe_501_so101.control.gripper_closing import ClosingProfile
    from scripts.sim2real.analyze.gripper_closing_check import build_end_to_end

    preset = CartesianZeroPreset()
    base = replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass))
    env = make_env(replace(preset, base=base))
    reference, script = build_end_to_end(ClosingProfile("position_zero", target_deg=0.), mass, True)
    try:
        obs, _ = env.reset(seed=68006)
        expected, _ = reference.reset(seed=68006)
        script.reset()
        for _ in range(200):
            obs, _, _, _, info = env.step(np.zeros(6))
            expected, *_ = reference.step(script(expected))
            np.testing.assert_allclose(obs, expected, atol=1e-7)
        assert info["is_success"]
        assert env.unwrapped.config.gripper.close_deg == 0
    finally:
        env.close()
        reference.close()


def test_shadow_reference_waits_for_actual_lift_release():
    """Late closure must not advance a fictitious reference while the guard holds the arm."""
    preset = CartesianZeroPreset()
    env = make_env(preset)
    try:
        env.reset(seed=68006)
        for _ in range(101):
            env.step(np.zeros(6))
        assert not env.env.env.gate.released
        assert env.reference is None
        for _ in range(20):
            env.step(np.zeros(6))
        assert env.env.env.gate.released
        assert env.reference is not None
    finally:
        env.close()

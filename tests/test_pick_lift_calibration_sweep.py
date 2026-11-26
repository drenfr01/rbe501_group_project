"""Checks for the evaluation-only calibrated pick-and-lift comparison."""

from dataclasses import asdict

import pytest

from scripts.sim2real.analyze.pick_lift_calibration import EpisodeResult, make_config, summarize


def test_only_mass_and_opt_in_calibration_change_between_conditions():
    """Do not silently change contact physics, action mapping, or task settings."""
    original = asdict(make_config(0.01, False))
    calibrated = asdict(make_config(0.01, True))
    assert original["sim"].pop("calibration_path") is None
    assert calibrated["sim"].pop("calibration_path") is not None
    assert original == calibrated
    assert original["object"]["side_m"] == 0.025


def test_summary_counts_final_success_not_transient_success():
    """Lifting briefly and then dropping the cube is not a successful episode."""
    episodes = [
        EpisodeResult(0, True, True, True, 0.06, 0.07, 0, False),
        EpisodeResult(1, False, True, True, 0.0, 0.06, 2, False),
    ]
    result = summarize("measured_stiffness_only", "scripted", 0.01, episodes)
    assert result.success_rate == 0.5
    assert result.mean_final_lift_cm == pytest.approx(3.0)
    assert result.lost_success_episodes == 1
    assert result.robot_table_contact_episodes == 1
    assert result.warning_episodes == 0


def test_empty_summary_is_rejected():
    """Missing evaluation data must never look like a passing result."""
    with pytest.raises(ValueError, match="episodes"):
        summarize("original", "scripted", 0.01, [])

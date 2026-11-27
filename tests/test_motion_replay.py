"""Experimental motion replay must require explicit calibrated, guarded operation."""

import pytest

from scripts.sim2real.analyze.replay_pick_lift import replay


def test_motion_replay_rejects_ambiguous_configuration(tmp_path):
    """Reject invalid options before creating an output directory or loading a policy."""
    output = tmp_path / "replay"
    with pytest.raises(ValueError, match="calibrated.*guarded"):
        replay("scripted", .01, 61000, True, output, motion_variant="gravity_yaw_depth4")
    assert not output.exists()

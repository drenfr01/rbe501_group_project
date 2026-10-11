"""Unit tests for experiment reporting helpers."""

import json
from pathlib import Path

import numpy as np
import pytest

from rbe_501_so101.rl.reporting import TrainingCurve, load_report

MONITOR_CSV = '#{"t_start": 0.0, "env_id": "SO101Reach-v0"}\nr,l,t\n-10.0,150,1.0\n-6.0,150,2.0\n-2.0,150,3.0\n'


@pytest.fixture
def monitor_path(tmp_path: Path) -> Path:
    """A three-episode SB3 Monitor CSV."""
    path = tmp_path / "train.monitor.csv"
    path.write_text(MONITOR_CSV)
    return path


def test_curve_uses_cumulative_episode_lengths(monitor_path):
    curve = TrainingCurve.from_monitor(monitor_path, "0")
    np.testing.assert_array_equal(curve.timesteps, [150, 300, 450])
    np.testing.assert_array_equal(curve.rewards, [-10.0, -6.0, -2.0])


def test_smoothed_is_trailing_mean(monitor_path):
    curve = TrainingCurve.from_monitor(monitor_path, "0")
    np.testing.assert_allclose(curve.smoothed(window=2), [-10.0, -8.0, -4.0])


def test_tail_mean_uses_last_fraction(monitor_path):
    curve = TrainingCurve.from_monitor(monitor_path, "0")
    assert curve.tail_mean(1 / 3) == pytest.approx(-2.0)


def test_load_report_skips_leading_log_lines(tmp_path):
    path = tmp_path / "report.raw"
    path.write_text("Using cuda device\n" + json.dumps({"passed": False}))
    assert load_report(path) == {"passed": False}

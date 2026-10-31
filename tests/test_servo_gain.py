"""Servo position gain: configured in HardwareConfig, passed to LeRobot at connect, recorded with results."""

from dataclasses import replace
from datetime import datetime

import pytest

from rbe_501_so101.config import HardwareConfig
from rbe_501_so101.hardware.bus import FakeJointBus, follower_config_kwargs
from rbe_501_so101.hardware.pose_check import PoseCheckRepository, run_suffix


def test_follower_config_carries_the_p_gain():
    kwargs = follower_config_kwargs(replace(HardwareConfig(), position_p_gain=32))
    assert kwargs["position_p_coefficient"] == 32
    assert kwargs["max_relative_target"] == HardwareConfig().max_relative_target_deg


def test_follower_config_is_accepted_by_lerobot():
    from lerobot.robots.so_follower import SO101FollowerConfig

    robot_cfg = SO101FollowerConfig(**follower_config_kwargs(replace(HardwareConfig(), position_p_gain=24)))
    assert robot_cfg.position_p_coefficient == 24


@pytest.mark.parametrize("gain", [0, 255, -1])
def test_out_of_range_gain_is_refused(gain):
    with pytest.raises(ValueError, match="P gain"):
        follower_config_kwargs(replace(HardwareConfig(), position_p_gain=gain))


def test_run_folders_name_the_gain_and_never_collide():
    first = run_suffix("rbe501_follower", 32, datetime(2026, 10, 31, 12, 0, 5))
    second = run_suffix("rbe501_follower", 32, datetime(2026, 10, 31, 12, 4, 0))
    assert first == "rbe501_follower_p32_120005"
    assert first != second


def test_settings_are_saved_next_to_results(tmp_path):
    repo = PoseCheckRepository(tmp_path / "results.json")
    settings = FakeJointBus(("shoulder_lift", "gripper")).read_settings()
    path = repo.save_settings(settings)
    assert path.parent == tmp_path
    assert repo.load_settings() == settings

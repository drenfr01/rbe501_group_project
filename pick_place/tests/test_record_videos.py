"""Tests for the simulation video recorder (scripts/record_videos.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_videos.py"
spec = importlib.util.spec_from_file_location("record_videos", SCRIPT)
videos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(videos)

SHOWCASE_1KG = 20_008
"""Episode where the scripted controller misses by 2.2 cm at 1 kg and feedforward places it."""


def test_tile_puts_frames_in_a_grid_row_by_row():
    frames = [np.full((48, 64, 3), k, dtype=np.uint8) for k in range(4)]
    grid = videos.tile(frames, columns=2)
    assert grid.shape == (96, 128, 3)
    assert grid[0, 64, 0] == 1 and grid[48, 0, 0] == 2


@pytest.fixture(scope="module")
def scripted_clip():
    """Scripted controller at 1 kg on the showcase episode."""
    return videos.rollout("scripted", 1.0, SHOWCASE_1KG)


def test_rollout_renders_one_labelled_frame_per_step(scripted_clip):
    assert len(scripted_clip.torque_nm) == 320
    assert len(scripted_clip.frames) == 320 + videos.HOLD_FRAMES  # plus the held result banner
    assert scripted_clip.frames[0].shape == (videos.HEIGHT, videos.WIDTH, 3)
    assert scripted_clip.frames[0].std() > 10  # a real image, not a blank buffer


def test_rollout_keeps_the_episode_outcome(scripted_clip):
    feedforward = videos.rollout("feedforward", 1.0, SHOWCASE_1KG)
    assert not scripted_clip.success and feedforward.success
    assert scripted_clip.error_m > feedforward.error_m


def test_rollout_can_pick_the_trained_policy_seed():
    """With training seed 4 at 1 kg, fixed places episode 20008 and random misses it."""
    fixed = videos.rollout("fixed", 1.0, SHOWCASE_1KG, policy_seed=4)
    random = videos.rollout("random", 1.0, SHOWCASE_1KG, policy_seed=4)
    assert fixed.success and not random.success
    assert fixed.policy_seed == 4


def test_torque_chart_matches_the_frame_height(scripted_clip):
    chart = videos.torque_chart(scripted_clip, step=100)
    assert chart.shape == (videos.HEIGHT, videos.CHART_WIDTH, 3)

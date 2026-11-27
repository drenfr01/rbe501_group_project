"""Shared hardware-session wiring: the real backend stack running end to end on the simulated bus."""

import numpy as np
import pytest

from rbe_501_so101.config import HardwareConfig, ReachTaskConfig, SimConfig, TrainConfig
from rbe_501_so101.envs.reach_env import SO101ReachEnv
from rbe_501_so101.envs.targets import SequenceTargetSampler
from rbe_501_so101.envs.types import ObservationLayout
from rbe_501_so101.hardware.session import build_real_backend, open_bus
from rbe_501_so101.hardware.sim_bus import SimJointBus
from rbe_501_so101.provenance import experiment_dir
from rbe_501_so101.reports import publish_figure
from rbe_501_so101.rl.evaluation import run_episode
from rbe_501_so101.rl.policies import make_baseline

SIM = SimConfig()
TARGET = np.array([0.15, 0.0, 0.6])
FAST_HW = HardwareConfig(home_ramp_s=0.5, settle_s=0.5, rest_ramp_s=0.5)


def test_open_bus_in_sim_returns_sim_bus_as_its_own_clock():
    bus, clock = open_bus(sim=True, hw=FAST_HW)
    assert isinstance(bus, SimJointBus) and clock is bus
    bus.close()


def _hold_error_mm(name: str) -> float:
    """Encoder-based holding error of a baseline driven through the real backend on the sim bus."""
    bus, clock = open_bus(sim=True, hw=FAST_HW)
    backend = build_real_backend(bus, clock, FAST_HW, home_deg=(*SIM.initial_deg[:5], ReachTaskConfig().gripper_hold_deg))
    env = SO101ReachEnv(backend=backend, sampler=SequenceTargetSampler((TARGET,)))
    policy = make_baseline(name, SIM, ReachTaskConfig(), ObservationLayout(n_arm=len(SIM.arm_joints)))
    try:
        stats = run_episode(env, policy, seed=0, window=TrainConfig().holding_window_steps)
    finally:
        env.close()
    return 1000.0 * float(np.mean(stats.holding_errors_m))


def test_real_backend_on_sim_bus_shows_sag_and_integral_corrects_it():
    ik_hold, integral = _hold_error_mm("ik_hold"), _hold_error_mm("integral")
    assert ik_hold > 3.0
    assert integral < ik_hold / 3


def test_experiment_dir_is_dated(tmp_path):
    path = experiment_dir("thing", base=tmp_path)
    assert path.parent == tmp_path and path.name.endswith("_thing") and path.name[:4].isdigit()


def test_publish_figure_copies_under_stable_name(tmp_path):
    src = tmp_path / "a.png"
    src.write_bytes(b"png")
    dest = publish_figure(src, "kp_sweep.png", reports_dir=tmp_path / "reports")
    assert dest == tmp_path / "reports" / "figures" / "kp_sweep.png"
    assert dest.read_bytes() == b"png"


def test_sim_bus_clips_relative_to_present_like_lerobot():
    bus, _ = open_bus(sim=True, hw=FAST_HW)
    present = bus.read_positions()
    key = "elbow_flex.pos"
    sent = bus.send_positions({key: present[key] + 30.0})
    assert sent[key] == pytest.approx(present[key] + FAST_HW.max_relative_target_deg)
    bus.close()


def test_sim_bus_does_not_clip_the_gripper_it_never_moves():
    bus, _ = open_bus(sim=True, hw=FAST_HW)
    key = "gripper.pos"
    far = bus.read_positions()[key] - 30.0
    assert bus.send_positions({key: far})[key] == pytest.approx(far)
    bus.close()

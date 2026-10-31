"""Phase 3 trial logging/repeatability and Phase 1 reference readings/offsets."""

import numpy as np
import pytest

from rbe_501_so101.config import HardwareConfig, IKConfig, JointMappingEntry, ReachTaskConfig, SafetyConfig, SimConfig
from rbe_501_so101.envs.reach_env import SO101ReachEnv
from rbe_501_so101.envs.targets import SequenceTargetSampler
from rbe_501_so101.envs.types import ObservationLayout
from rbe_501_so101.hardware.bus import FakeJointBus
from rbe_501_so101.hardware.clock import FakeClock
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.real_backend import LeRobotSo101Backend
from rbe_501_so101.hardware.references import ReferenceReading, ReferenceRepository, offset_from_reference
from rbe_501_so101.hardware.safety import SafetyMonitor
from rbe_501_so101.hardware.trials import (
    RepeatabilityStats,
    TargetSummary,
    TrialRepository,
    TrialSummary,
    load_targets,
    repeatability,
    run_logged_episode,
    summarize_trials,
    trial_readme,
    trial_schedule,
)
from rbe_501_so101.kinematics import PositionIKSolver
from rbe_501_so101.rl.policies import IKHoldPolicy

SIM = SimConfig()
TARGET = np.array([0.15, 0.0, 0.6])


def test_repeatability_reports_spread_about_the_centroid():
    finals = np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]])
    stats = repeatability(finals)
    assert stats.mean_radius_mm == pytest.approx(1.0)
    assert stats.max_radius_mm == pytest.approx(1.0)
    assert stats.n == 2


@pytest.fixture
def logged_episode(raw_model):
    """One logged IK-hold episode on a fake arm with sag."""
    kinematics = PositionIKSolver.from_config(SIM, IKConfig())
    bus = FakeJointBus(SIM.joints, sag_deg={"shoulder_lift": 1.0})
    joint_map = JointMap.from_model(raw_model, SIM, HardwareConfig())
    safety = SafetyMonitor.from_model(raw_model, SIM, SafetyConfig())
    backend = LeRobotSo101Backend(bus, joint_map, safety, kinematics, raw_model, SIM, HardwareConfig(), FakeClock())
    env = SO101ReachEnv(backend=backend, sampler=SequenceTargetSampler((TARGET,)))
    policy = IKHoldPolicy(kinematics, ReachTaskConfig(), ObservationLayout(n_arm=len(SIM.arm_joints)))
    record = run_logged_episode(env, policy, "ik_hold", target_index=0, repeat=0, seed=0)
    env.close()
    return record


def test_logged_episode_has_every_step(logged_episode):
    assert len(logged_episode.steps) == ReachTaskConfig().max_episode_steps
    assert logged_episode.controller == "ik_hold"
    assert all(np.isfinite(s.error_m) for s in logged_episode.steps)


def test_trial_repository_round_trips_episode(tmp_path, logged_episode):
    repo = TrialRepository(tmp_path)
    path = repo.save_episode(logged_episode)
    assert repo.load_episode(path) == logged_episode


def test_summarize_trials_groups_by_target(logged_episode):
    summary = summarize_trials([logged_episode, logged_episode], holding_window=10)
    assert summary.n_episodes == 2
    assert summary.per_target[0].repeatability.n == 2
    assert summary.per_target[0].repeatability.max_radius_mm == pytest.approx(0.0)


def test_offset_from_reference_makes_the_map_hit_the_reference():
    offset = offset_from_reference(lerobot_deg=12.0, sim_reference_rad=0.5, sign=-1.0)
    entries = (JointMappingEntry(), JointMappingEntry(-1.0, offset), *(JointMappingEntry() for _ in range(3)))
    joint_map = JointMap(SIM.arm_joints, SIM.gripper_joint, entries, (0.0, 100.0), (0.0, 1.0))
    reading = {pos_key(j): 0.0 for j in SIM.joints} | {pos_key("shoulder_lift"): 12.0}
    assert joint_map.to_sim(reading)[1] == pytest.approx(0.5)


def test_reference_repository_appends_and_loads(tmp_path):
    repo = ReferenceRepository(tmp_path / "refs.json")
    first = ReferenceReading("upper arm level", {"shoulder_lift.pos": 3.0}, (0.0, 3.0))
    repo.append(first)
    repo.append(ReferenceReading("forearm level", {"elbow_flex.pos": -2.0}, (0.0, -2.0)))
    loaded = repo.load()
    assert len(loaded) == 2 and loaded[0] == first


def test_load_targets_reads_sim_world_points(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text('{"frame": "world", "targets": [[0.15, 0.0, 0.6], [0.2, 0.05, 0.58]]}')
    targets = load_targets(path)
    assert len(targets) == 2
    np.testing.assert_allclose(targets[1], [0.2, 0.05, 0.58])


def test_trial_schedule_repeats_each_target_in_order():
    targets = (np.zeros(3), np.ones(3))
    schedule = trial_schedule(targets, repeats=2)
    assert [(t.target_index, t.repeat) for t in schedule] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    np.testing.assert_array_equal(schedule[2].target, np.ones(3))


def test_trial_readme_lists_overall_and_per_target_rows():
    spread = RepeatabilityStats(10, 0.4, 0.9)
    summary = TrialSummary("ik_hold", 20, 11.0, 12.5, 3, {0: TargetSummary(0, 10.0, 11.0, spread), 1: TargetSummary(1, 12.0, 14.0, spread)})
    text = trial_readme(summary, aborted="operator abort key pressed")
    assert "ik_hold" in text and "11.00" in text and "| 1 | 12.00 | 14.00 | 10 | 0.40 | 0.90 |" in text
    assert "operator abort" in text and "overruns: 3" in text

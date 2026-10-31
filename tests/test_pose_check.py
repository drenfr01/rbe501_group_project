"""Pose check: drive the arm to poses defined in sim degrees and compare with what it reads."""

import numpy as np
import pytest

from rbe_501_so101.config import CheckPose, HardwareConfig, PoseCheckConfig, SimConfig
from rbe_501_so101.hardware.bus import FakeJointBus
from rbe_501_so101.hardware.clock import FakeClock
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.pose_check import (
    PoseCheckRepository,
    PoseCheckRunner,
    plan_path,
    preflight,
    segment_points,
)
from rbe_501_so101.hardware.safety import SafetyAbort, SafetyMonitor

SIM = SimConfig()
HW = HardwareConfig()
FAST = PoseCheckConfig(settle_s=0.1, n_reads=3, ramp_speed_deg_s=120.0)
INTO_TABLE = CheckPose("into_table", (0.0, 90.0, 0.0, 0.0, 0.0), "gripper on the floor")


@pytest.fixture
def safety(raw_model):
    """Safety monitor on the scene model."""
    return SafetyMonitor.from_model(raw_model, SIM)


@pytest.fixture
def joint_map(raw_model):
    """Default (Phase 1) mapping."""
    return JointMap.from_model(raw_model, SIM, HW)


def _q(arm_deg: tuple[float, ...]) -> np.ndarray:
    """Full sim joint vector (gripper closed) from arm degrees."""
    return np.append(np.deg2rad(arm_deg), 0.0)


def test_segment_points_never_step_more_than_the_limit():
    points = segment_points(_q((0.0,) * 5), _q((0.0, 10.0, -25.0, 0.0, 0.0)), np.deg2rad(1.0))
    steps = np.abs(np.diff(points, axis=0)).max(axis=1)
    assert np.all(steps <= np.deg2rad(1.0) + 1e-12)
    np.testing.assert_allclose(points[-1], _q((0.0, 10.0, -25.0, 0.0, 0.0)))


def test_plan_returns_to_the_start():
    start = _q((0.0, -60.0, 60.0, 0.0, 0.0))
    path = plan_path(start, (_q((0.0,) * 5), _q((45.0, 0.0, 0.0, 0.0, 0.0))))
    assert len(path) == 4
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], start)


def test_preflight_passes_the_configured_poses_from_sim_zero(safety):
    poses = tuple(_q(p.sim_deg) for p in PoseCheckConfig().poses)
    assert preflight(safety, plan_path(_q((0.0,) * 5), poses), PoseCheckConfig()) > 0.0


def test_preflight_refuses_a_pose_inside_the_table(safety):
    path = plan_path(_q((0.0,) * 5), (_q(INTO_TABLE.sim_deg),))
    with pytest.raises(SafetyAbort, match="table"):
        preflight(safety, path, PoseCheckConfig())


def test_preflight_tolerates_a_low_start_but_not_going_lower(safety):
    low_start = _q((0.0, 60.0, 0.0, 0.0, 0.0))
    preflight(safety, plan_path(low_start, (_q((0.0,) * 5),)), PoseCheckConfig())
    with pytest.raises(SafetyAbort):
        preflight(safety, plan_path(low_start, (_q(INTO_TABLE.sim_deg),)), PoseCheckConfig())


def test_runner_commands_the_mapped_lerobot_goal_and_keeps_the_gripper(joint_map):
    bus = FakeJointBus(SIM.joints, start_deg={"gripper": 33.0}, sag_deg={"elbow_flex": 2.0})
    runner = PoseCheckRunner(bus, FakeClock(), joint_map, FAST, SIM.joints)
    pose = PoseCheckConfig().poses[0]
    runner.start()
    result = runner.visit(pose)
    expected = joint_map.to_lerobot(_q(pose.sim_deg))
    assert result.lerobot_goal_deg[:5] == pytest.approx([expected[pos_key(j)] for j in SIM.arm_joints])
    assert result.lerobot_goal_deg[5] == pytest.approx(33.0)
    assert result.measured_deg[2] == pytest.approx(result.lerobot_goal_deg[2] - 2.0)
    assert result.measured_sim_deg[2] == pytest.approx(pose.sim_deg[2] - 2.0)
    assert result.clamped is False


def test_runner_finishes_at_the_start_with_torque_off(joint_map):
    bus = FakeJointBus(SIM.joints, start_deg={"shoulder_lift": -80.0})
    runner = PoseCheckRunner(bus, FakeClock(), joint_map, FAST, SIM.joints)
    runner.start()
    runner.visit(PoseCheckConfig().poses[0])
    runner.finish()
    assert bus.sent[-1][pos_key("shoulder_lift")] == pytest.approx(-80.0)
    assert bus.torque_log == [True, False]


def test_repository_round_trip(joint_map, tmp_path):
    bus = FakeJointBus(SIM.joints)
    runner = PoseCheckRunner(bus, FakeClock(), joint_map, FAST, SIM.joints)
    runner.start()
    results = (runner.visit(PoseCheckConfig().poses[0]).with_note("forearm level"),)
    repo = PoseCheckRepository(tmp_path / "results.json")
    repo.save(results)
    assert repo.load() == results

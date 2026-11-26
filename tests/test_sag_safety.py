"""Sag-check safety: gravity-aware approach sides, a raised waypoint, the path check, and per-run folders."""

from datetime import datetime

import numpy as np
import pytest

from rbe_501_so101.config import HardwareConfig, PathCheckConfig, SagCheckConfig, SimConfig
from rbe_501_so101.hardware.bus import FakeJointBus
from rbe_501_so101.hardware.clock import FakeClock
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.path_safety import preflight
from rbe_501_so101.hardware.safety import SafetyAbort, SafetyMonitor
from rbe_501_so101.hardware.sag import RecordedPose
from rbe_501_so101.hardware.sag_procedure import (
    GravityAwareOffsets,
    SagCheckRunner,
    UniformOffsets,
    with_approach_offset,
)
from rbe_501_so101.kinematics.gravity import GravityTorqueModel
from rbe_501_so101.provenance import timestamped_suffix

SIM = SimConfig()
JOINTS = SIM.joints
FAST = SagCheckConfig(settle_s=0.1, n_reads=2, ramp_speed_deg_s=120.0)
ARM_KEYS = [pos_key(j) for j in SIM.arm_joints]


@pytest.fixture
def joint_map(raw_model):
    """Default (Phase 1) mapping."""
    return JointMap.from_model(raw_model, SIM, HardwareConfig())


@pytest.fixture
def gravity_offsets(raw_model, joint_map):
    """Approach offsets that put "above" on the side gravity pulls away from."""
    return GravityAwareOffsets(joint_map, GravityTorqueModel.from_config(SIM), FAST.approach_offset_deg)


def _goal(joint_map: JointMap, sim_arm_deg: tuple[float, ...]) -> dict[str, float]:
    """LeRobot goal for a sim arm pose (gripper closed)."""
    return joint_map.to_lerobot(np.append(np.deg2rad(sim_arm_deg), 0.0))


def _holding_torque(joint_map: JointMap, goal: dict[str, float]) -> np.ndarray:
    """Torque each arm actuator needs to hold `goal` against gravity."""
    return GravityTorqueModel.from_config(SIM).torque(joint_map.to_sim(goal))


def test_uniform_offsets_shift_every_arm_joint_the_same_way():
    offsets = UniformOffsets(5.0, JOINTS)({pos_key(j): 0.0 for j in JOINTS})
    assert [offsets[k] for k in ARM_KEYS] == [5.0] * 5
    assert offsets[pos_key("gripper")] == 0.0


@pytest.mark.parametrize("sim_pose", [(0.0, 30.0, -30.0, 0.0, 0.0), (0.0, -60.0, 60.0, 30.0, 0.0)])
def test_gravity_aware_above_side_lifts_against_gravity(joint_map, gravity_offsets, sim_pose):
    goal = _goal(joint_map, sim_pose)
    offsets = gravity_offsets(goal)
    tau = _holding_torque(joint_map, goal)
    loaded = np.abs(tau) > SagCheckConfig().gravity_torque_floor_nm
    sim_dir = joint_map.arm_delta_to_sim(np.array([offsets[k] for k in ARM_KEYS]))
    assert np.all(np.sign(sim_dir[loaded]) == np.sign(tau[loaded]))
    assert np.allclose(np.abs([offsets[k] for k in ARM_KEYS]), FAST.approach_offset_deg)
    assert offsets[pos_key("gripper")] == 0.0


def test_plan_visits_waypoint_then_both_approach_sides_then_rest(joint_map, gravity_offsets):
    waypoint = _goal(joint_map, (0.0,) * 5)
    runner = SagCheckRunner(FakeJointBus(JOINTS), FakeClock(), FAST, JOINTS, gravity_offsets, waypoint)
    pose = RecordedPose(0, tuple(_goal(joint_map, (0.0, 30.0, -30.0, 0.0, 0.0)).values()))
    rest = {pos_key(j): 0.0 for j in JOINTS}
    plan = runner.plan((pose, pose), rest)
    goal = runner.goal_dict(pose.positions_deg)
    above, below = runner.approach_point(goal, "above"), runner.approach_point(goal, "below")
    one_pose = [runner.waypoint_for(goal), above, goal, below, goal]
    assert plan == [*one_pose, *one_pose, runner.waypoint_for(rest), rest]


def test_run_sends_exactly_the_plan_as_ramp_ends(joint_map, gravity_offsets):
    waypoint = _goal(joint_map, (0.0,) * 5)
    bus = FakeJointBus(JOINTS)
    runner = SagCheckRunner(bus, FakeClock(), FAST, JOINTS, gravity_offsets, waypoint)
    poses = (RecordedPose(0, tuple(_goal(joint_map, (0.0, 30.0, -30.0, 0.0, 0.0)).values())),)
    rest = bus.read_positions()
    runner.run(poses)
    runner.finish(rest)
    for target in runner.plan(poses, rest):
        assert target in bus.sent


def test_samples_record_the_offsets_used(joint_map, gravity_offsets):
    runner = SagCheckRunner(FakeJointBus(JOINTS), FakeClock(), FAST, JOINTS, gravity_offsets)
    pose = RecordedPose(0, tuple(_goal(joint_map, (0.0, 30.0, -30.0, 0.0, 0.0)).values()))
    above, below = runner.run((pose,))
    assert above.approach_offset_deg == pytest.approx(tuple(-v for v in below.approach_offset_deg))
    assert above.approach_offset_deg == pytest.approx(tuple(gravity_offsets(runner.goal_dict(pose.positions_deg)).values()))


def test_path_check_refuses_a_plan_that_dips_into_the_table(raw_model, joint_map, gravity_offsets):
    runner = SagCheckRunner(FakeJointBus(JOINTS), FakeClock(), FAST, JOINTS, gravity_offsets)
    low = RecordedPose(0, tuple(_goal(joint_map, (0.0, 90.0, 0.0, 0.0, 0.0)).values()))
    start = _goal(joint_map, (0.0,) * 5)
    path = [joint_map.to_sim(g) for g in [start, *runner.plan((low,), start)]]
    with pytest.raises(SafetyAbort, match="table"):
        preflight(SafetyMonitor.from_model(raw_model, SIM), path, PathCheckConfig())


def test_path_check_passes_a_high_plan_with_waypoint(raw_model, joint_map, gravity_offsets):
    waypoint = _goal(joint_map, (0.0,) * 5)
    runner = SagCheckRunner(FakeJointBus(JOINTS), FakeClock(), FAST, JOINTS, gravity_offsets, waypoint)
    poses = tuple(
        RecordedPose(i, tuple(_goal(joint_map, p).values()))
        for i, p in enumerate([(0.0, 20.0, -20.0, 0.0, 0.0), (30.0, -20.0, 10.0, 20.0, 0.0)])
    )
    path = [joint_map.to_sim(g) for g in [waypoint, *runner.plan(poses, waypoint)]]
    assert preflight(SafetyMonitor.from_model(raw_model, SIM), path, PathCheckConfig()) > 0.0


def test_timestamped_suffix_names_gain_and_time():
    assert timestamped_suffix("rbe501_follower", 24, datetime(2026, 10, 31, 15, 2, 9)) == "rbe501_follower_p24_150209"


def test_approach_offset_within_range_is_applied():
    """A larger approach offset (to sweep the whole friction band) replaces the default."""
    cfg = with_approach_offset(SagCheckConfig(), 8.0)
    assert cfg.approach_offset_deg == 8.0


@pytest.mark.parametrize("offset", [0.0, -3.0, SagCheckConfig().approach_offset_range_deg[1] + 1.0])
def test_approach_offset_outside_range_is_refused(offset):
    """Zero, negative, or very large offsets are refused before anything moves."""
    with pytest.raises(ValueError, match="approach offset"):
        with_approach_offset(SagCheckConfig(), offset)

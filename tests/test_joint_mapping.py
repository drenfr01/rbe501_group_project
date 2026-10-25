"""JointMap: LeRobot degrees / gripper percent to sim radians and back."""

import numpy as np
import pytest

from rbe_501_so101.config import HardwareConfig, JointMappingEntry, SimConfig
from rbe_501_so101.hardware.joint_mapping import JointMap

SIM = SimConfig()
GRIPPER_RANGE_RAD = (-0.17, 1.75)


def _map(entries: tuple[JointMappingEntry, ...] | None = None) -> JointMap:
    """Joint map with identity arm entries unless overridden."""
    hw = HardwareConfig() if entries is None else HardwareConfig(joint_map=entries)
    return JointMap(SIM.arm_joints, SIM.gripper_joint, hw.joint_map, hw.gripper_percent_range, GRIPPER_RANGE_RAD)


def _reading(values: list[float]) -> dict[str, float]:
    """LeRobot-style `<joint>.pos` dictionary in SimConfig joint order."""
    return {f"{name}.pos": v for name, v in zip(SIM.joints, values, strict=True)}


def test_round_trip_returns_the_reading():
    joint_map = _map((JointMappingEntry(-1.0, 0.1), *(JointMappingEntry(1.0, -0.2) for _ in range(4))))
    reading = _reading([10.0, -35.5, 60.0, 5.0, -90.0, 42.0])
    back = joint_map.to_lerobot(joint_map.to_sim(reading))
    assert back.keys() == reading.keys()
    np.testing.assert_allclose(list(back.values()), list(reading.values()))


def test_sign_flips_direction_and_offset_shifts_zero():
    flipped = _map((JointMappingEntry(-1.0, 0.0), *(JointMappingEntry() for _ in range(4))))
    shifted = _map((JointMappingEntry(1.0, 0.3), *(JointMappingEntry() for _ in range(4))))
    reading = _reading([30.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert flipped.to_sim(reading)[0] == pytest.approx(-np.deg2rad(30.0))
    assert shifted.to_sim(_reading([0.0] * 6))[0] == pytest.approx(0.3)


def test_gripper_ends_map_to_sim_range_ends():
    joint_map = _map()
    assert joint_map.to_sim(_reading([0.0] * 5 + [0.0]))[-1] == pytest.approx(GRIPPER_RANGE_RAD[0])
    assert joint_map.to_sim(_reading([0.0] * 5 + [100.0]))[-1] == pytest.approx(GRIPPER_RANGE_RAD[1])


def test_missing_joint_key_raises_clear_error():
    reading = _reading([0.0] * 6)
    del reading["elbow_flex.pos"]
    with pytest.raises(KeyError, match="elbow_flex.pos"):
        _map().to_sim(reading)


def test_arm_limits_convert_and_stay_ordered_under_sign_flip():
    joint_map = _map((JointMappingEntry(-1.0, 0.0), *(JointMappingEntry() for _ in range(4))))
    limits = joint_map.arm_limits_to_sim({name: (-90.0, 45.0) for name in SIM.arm_joints})
    assert limits.lower[0] == pytest.approx(-np.deg2rad(45.0))
    assert limits.upper[0] == pytest.approx(np.deg2rad(90.0))
    assert np.all(limits.lower < limits.upper)


def test_from_model_reads_gripper_range(raw_model):
    joint_map = JointMap.from_model(raw_model, SIM, HardwareConfig())
    expected = raw_model.joint(SIM.gripper_joint).range
    assert joint_map.to_sim(_reading([0.0] * 5 + [100.0]))[-1] == pytest.approx(expected[1])

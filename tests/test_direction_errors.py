"""Servo error split by arrival direction (with or against gravity), and the hardware-day figures."""

import numpy as np
import pytest

from rbe_501_so101.config import HardwareConfig, JointMappingEntry, SimConfig
from rbe_501_so101.hardware.direction_errors import (
    DirectionalError,
    directional_errors,
    opposite_arrival_bands,
    pull_sign,
    visit_errors,
)
from rbe_501_so101.hardware.figures import (
    BandSeries,
    plot_direction_errors,
    plot_friction_bands,
    plot_gain_sweep,
    plot_stiffness,
)
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.pose_check import PoseCheckResult
from rbe_501_so101.hardware.sag import JointComplianceFit, SagSample, SagSession
from rbe_501_so101.kinematics.gravity import GravityTorqueModel

SIM = SimConfig()
IDENTITY_HW = HardwareConfig(joint_map=tuple(JointMappingEntry() for _ in SIM.arm_joints))
GOAL_DEG = (0.0, 20.0, -10.0, 10.0, 0.0, 30.0)
WITH_DEG, AGAINST_DEG = 0.5, 4.0
LOADED = ("shoulder_lift", "elbow_flex")


@pytest.fixture
def joint_map(raw_model):
    """Identity mapping, so LeRobot and sim degrees coincide."""
    return JointMap.from_model(raw_model, SIM, IDENTITY_HW)


@pytest.fixture
def gravity():
    """Nominal-model gravity torque."""
    return GravityTorqueModel.from_config(SIM)


def _pull(gravity) -> np.ndarray:
    """Sim direction gravity pulls each arm joint at the goal."""
    return pull_sign(gravity.torque(np.append(np.deg2rad(GOAL_DEG[:5]), 0.0)))


def _sample(approach: str, error_along_pull: float, start_side: np.ndarray, pull: np.ndarray) -> SagSample:
    """A sample that started on `start_side` of the goal and settled `error_along_pull` along gravity."""
    measured = np.array(GOAL_DEG)
    measured[:5] += error_along_pull * pull
    offsets = (*(3.0 * start_side), 0.0)
    zeros = (0.0,) * 6
    return SagSample(0, approach, GOAL_DEG, tuple(measured), zeros, (), approach_offset_deg=offsets)


def _session(gravity) -> SagSession:
    """One pose: "above" starts up-gravity (arrives with it), "below" starts down-gravity."""
    pull = _pull(gravity)
    samples = (_sample("above", WITH_DEG, -pull, pull), _sample("below", AGAINST_DEG, pull, pull))
    return SagSession("test", SIM.joints, (), samples)


def test_pull_sign_is_opposite_the_holding_torque():
    """Gravity pulls the way the actuator is not pushing."""
    assert pull_sign(np.array([0.4, -0.2, 0.0])).tolist() == [-1.0, 1.0, 0.0]


def test_directional_errors_split_by_arrival_side(joint_map, gravity):
    """The sample that started up-gravity is the with-gravity arrival."""
    errors = directional_errors(_session(gravity), joint_map, gravity, LOADED)
    assert [e.joint for e in errors] == list(LOADED)
    for e in errors:
        assert e.with_gravity_deg == pytest.approx(WITH_DEG)
        assert e.against_gravity_deg == pytest.approx(AGAINST_DEG)
        assert e.band_deg == pytest.approx(AGAINST_DEG - WITH_DEG)
        assert e.gravity_load_nm > 0


def test_directional_errors_skip_unloaded_joints(joint_map, gravity):
    """Pan carries no gravity load, so it has no with/against split."""
    errors = directional_errors(_session(gravity), joint_map, gravity, ("shoulder_pan", "shoulder_lift"))
    assert [e.joint for e in errors] == ["shoulder_lift"]


def _visit(name: str, goal: tuple[float, ...], error: tuple[float, ...]) -> PoseCheckResult:
    """A pose-check visit with the given sim goal and sim error (arm joints)."""
    measured = tuple(g + e for g, e in zip(goal, error, strict=True))
    return PoseCheckResult(name, "", goal, goal, goal, measured, (0.0,) * 5, measured, False)


def test_visit_errors_use_the_move_direction(gravity):
    """Arriving at the L pose from behind (lift increasing) moves the shoulder with gravity."""
    pull = _pull_at(gravity, (0.0,) * 5)
    lift = 1
    behind = np.zeros(5)
    behind[lift] = -40.0 * pull[lift]
    ahead = -behind
    results = (
        _visit("L_pose", (0.0,) * 5, tuple(0.5 * pull)),
        _visit("lean", tuple(ahead), (0.0,) * 5),
        _visit("L_pose", (0.0,) * 5, tuple(4.0 * pull)),
    )
    visits = visit_errors(results, tuple(behind), gravity, SIM.arm_joints)
    lift_visits = [v for v in visits if v.joint == "shoulder_lift" and v.name == "L_pose"]
    assert [v.arrived_with_gravity for v in lift_visits] == [True, False]
    assert [v.error_deg for v in lift_visits] == pytest.approx([0.5, 4.0])
    assert opposite_arrival_bands(visits)["shoulder_lift"] == pytest.approx(3.5)


def _pull_at(gravity, sim_deg: tuple[float, ...]) -> np.ndarray:
    """Gravity pull direction at a sim pose (arm joints, degrees)."""
    return pull_sign(gravity.torque(np.append(np.deg2rad(sim_deg), 0.0)))


def _errors() -> tuple[DirectionalError, ...]:
    """Two loaded joints at two poses."""
    return tuple(
        DirectionalError(pid, joint, load, 0.3, 3.5) for pid, load in ((0, 0.2), (1, 0.5)) for joint in LOADED
    )


def test_figures_are_written(tmp_path):
    """Every hardware-day figure renders to a PNG."""
    bands = [BandSeries("8 deg", {"shoulder_lift": 3.1, "elbow_flex": 3.6}, {"shoulder_lift": (2.5, 3.8)})]
    fit = JointComplianceFit("shoulder_lift", True, 5, 0.3, 0.1, (0.09, 0.11), 10.0, 9.0, 11.0)
    paths = [tmp_path / f"{n}.png" for n in ("bands", "directions", "stiffness", "gains")]
    plot_friction_bands(bands, LOADED, paths[0])
    plot_direction_errors({"8 deg": _errors()}, {"shoulder_lift": 10.0, "elbow_flex": 11.0}, LOADED, paths[1])
    plot_stiffness((fit,), {"shoulder_lift": 20.0}, paths[2])
    plot_gain_sweep((16, 24, 32), {"shoulder_lift": (7.7, 5.0, 3.6)}, paths[3])
    assert all(p.stat().st_size > 0 for p in paths)


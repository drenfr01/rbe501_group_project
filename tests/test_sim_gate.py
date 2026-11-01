"""Calibrated-sim gate: replay hardware procedures in sim and compare with the real arm."""

import numpy as np
import pytest

from rbe_501_so101.config import (
    DEFAULT_CHECK_POSES,
    CalibratedSimGateConfig,
    HardwareConfig,
    JointMappingEntry,
    PoseCheckConfig,
    SagCheckConfig,
    ServoEffectsConfig,
    SimConfig,
)
from rbe_501_so101.envs.calibration import ActuatorCalibration, JointActuatorCalibration
from rbe_501_so101.hardware.direction_errors import DirectionalError, VisitError, directional_errors
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.sag import SagSample, SagSession
from rbe_501_so101.hardware.sim_gate import GateResult, directional_rows, visit_rows
from rbe_501_so101.hardware.sim_replay import ReplaySetup, replay_pose_check, replay_sag_session
from rbe_501_so101.kinematics.gravity import GravityTorqueModel

SIM = SimConfig()
IDENTITY_HW = HardwareConfig(joint_map=tuple(JointMappingEntry() for _ in SIM.arm_joints))
FAST = SagCheckConfig(settle_s=1.0, n_reads=2, ramp_speed_deg_s=60.0)
BAND_DEG = 1.5
OFFSET_DEG = 8.0
GOALS = ((0.0, 20.0, -10.0, 10.0, 0.0, 30.0), (10.0, 35.0, -25.0, 5.0, 0.0, 30.0))
LOADED = ("shoulder_lift", "elbow_flex")


@pytest.fixture
def joint_map(raw_model):
    """Identity mapping."""
    return JointMap.from_model(raw_model, SIM, IDENTITY_HW)


def _calibrated_sim(tmp_path) -> SimConfig:
    """Sim config whose calibration has the measured stiffness and a 1.5° band on every arm joint."""
    entries = tuple(
        JointActuatorCalibration(j, kp_nm_per_rad=11.0, stiction_half_band_rad=np.deg2rad(BAND_DEG),
                                 stiction_load_slope_rad_per_nm=0.0)
        for j in SIM.arm_joints
    )
    path = tmp_path / "cal.json"
    ActuatorCalibration(entries).save(path)
    return SimConfig(calibration_path=path)


def _real_session() -> SagSession:
    """Two poses; "above" starts up-gravity on every loaded joint (the measured values are irrelevant here)."""
    gravity = GravityTorqueModel.from_config(SIM)
    samples = []
    for pid, goal in enumerate(GOALS):
        up = np.sign(gravity.torque(np.deg2rad(goal)))
        up[[0, 4]] = 1.0
        for approach, sign in (("above", 1.0), ("below", -1.0)):
            offsets = (*(sign * OFFSET_DEG * up), 0.0)
            samples.append(SagSample(pid, approach, goal, goal, (0.0,) * 6, (), approach_offset_deg=offsets))
    return SagSession("real", SIM.joints, (), tuple(samples))


def _setup(tmp_path, joint_map, stiction: bool) -> ReplaySetup:
    """Replay on the calibrated sim, starting at the task home pose."""
    return ReplaySetup(
        _calibrated_sim(tmp_path), joint_map, ServoEffectsConfig(stiction=stiction), start_sim_deg=SIM.initial_deg
    )


def test_replay_with_stiction_reproduces_a_direction_dependent_stop(tmp_path, joint_map):
    """With the band, arriving against gravity stops about two bands further from the goal than with it."""
    sim = replay_sag_session(_real_session(), _setup(tmp_path, joint_map, True), FAST, waypoint_sim_deg=None)
    errors = directional_errors(sim, joint_map, GravityTorqueModel.from_config(SIM), LOADED)
    bands = [e.band_deg for e in errors]
    assert np.mean(bands) == pytest.approx(2 * BAND_DEG, rel=0.3)


def test_replay_without_stiction_lands_on_the_same_spot_from_both_sides(tmp_path, joint_map):
    """MuJoCo's spring alone has no lasting direction dependence."""
    sim = replay_sag_session(_real_session(), _setup(tmp_path, joint_map, False), FAST, waypoint_sim_deg=None)
    errors = directional_errors(sim, joint_map, GravityTorqueModel.from_config(SIM), LOADED)
    assert max(abs(e.band_deg) for e in errors) < 0.3


def test_pose_check_replay_visits_the_same_poses(tmp_path, joint_map):
    """The replay walks the real run's pose names in order."""
    poses = {p.name: p for p in DEFAULT_CHECK_POSES}
    names = ("L_pose", "lean_forward_30", "L_pose")
    cfg = PoseCheckConfig(settle_s=0.3, n_reads=2, ramp_speed_deg_s=60.0)
    results = replay_pose_check(names, _setup(tmp_path, joint_map, True), cfg, poses)
    assert [r.name for r in results] == list(names)


def _errors(with_deg: float, against_deg: float) -> tuple[DirectionalError, ...]:
    """One pose per loaded joint."""
    return tuple(DirectionalError(0, j, 0.3, with_deg, against_deg) for j in LOADED)


def test_directional_rows_pass_within_tolerance():
    """Per joint and direction: mean absolute real-vs-sim difference against the tolerance."""
    tol = CalibratedSimGateConfig().tolerance_deg
    rows = directional_rows("8°", _errors(0.2, 4.0), _errors(0.2 + 0.5 * tol, 4.0 + 2 * tol), tol)
    by = {(r.joint, r.direction): r for r in rows}
    assert by[("shoulder_lift", "with gravity")].passed
    assert not by[("shoulder_lift", "against gravity")].passed
    assert by[("shoulder_lift", "against gravity")].mean_abs_diff_deg == pytest.approx(2 * tol)
    assert not GateResult(tuple(rows), tol).passed


def test_visit_rows_compare_the_same_visit():
    """Pose-check visits are matched by visit index and joint."""
    real = (VisitError(0, "L_pose", "shoulder_lift", 0.5, 4.0, False),)
    sim = (VisitError(0, "L_pose", "shoulder_lift", 0.5, 3.5, False),)
    (row,) = visit_rows("pose check", real, sim, 1.0)
    assert row.direction == "against gravity" and row.passed
    assert row.mean_abs_diff_deg == pytest.approx(0.5)

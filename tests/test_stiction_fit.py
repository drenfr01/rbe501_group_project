"""Stiction (friction band) fit from a sag session, and what the calibration file records."""

from dataclasses import replace

import numpy as np
import pytest

from rbe_501_so101.config import HardwareConfig, JointMappingEntry, SagFitConfig, SimConfig
from rbe_501_so101.envs.calibration import ActuatorCalibration
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.sag import SagSample, SagSession, fit_stiction
from rbe_501_so101.hardware.sag_analysis import analyze_session, calibration_from_analysis
from rbe_501_so101.kinematics.gravity import GravityTorqueModel

SIM = SimConfig()
IDENTITY_HW = HardwareConfig(joint_map=tuple(JointMappingEntry() for _ in SIM.arm_joints))
POSES_DEG = (
    (0.0, -20.0, 40.0, 15.0, 0.0, 40.0),
    (10.0, 0.0, 20.0, 40.0, 30.0, 40.0),
    (-15.0, 40.0, -20.0, -30.0, -40.0, 40.0),
    (5.0, 55.0, -50.0, 0.0, 20.0, 40.0),
)
KP = np.array([20.0, 10.0, 11.0, 12.0, 20.0])
BAND0_DEG = np.array([1.0, 1.0, 1.2, 0.8, 0.9])
SLOPE_DEG_PER_NM = 2.0
LARGE_OFFSET_DEG = 8.0
SMALL_OFFSET_DEG = 3.0


def _tau(goal: tuple[float, ...]) -> np.ndarray:
    """Holding torque at a goal (identity mapping, so LeRobot = sim degrees)."""
    return GravityTorqueModel.from_config(SIM).torque(np.deg2rad(goal))


def _session(offset_deg: float) -> SagSession:
    """Spring sag plus a load-dependent friction band on shoulder lift, constant bands elsewhere."""
    samples = []
    for pid, goal in enumerate(POSES_DEG):
        tau = _tau(goal)
        sag = np.rad2deg(tau / KP)
        half = BAND0_DEG.copy()
        half[1] += SLOPE_DEG_PER_NM * abs(tau[1])
        for approach, sign in (("above", 1.0), ("below", -1.0)):
            measured = np.array(goal)
            measured[:5] += -sag + sign * half
            offsets = (*(sign * offset_deg * np.ones(5)), 0.0)
            samples.append(SagSample(pid, approach, goal, tuple(measured), (0.0,) * 6, (), approach_offset_deg=offsets))
    return SagSession("test", SIM.joints, (), tuple(samples))


@pytest.fixture
def joint_map(raw_model):
    """Identity mapping."""
    return JointMap.from_model(raw_model, SIM, IDENTITY_HW)


def test_stiction_fit_recovers_band_and_load_slope(joint_map, raw_model):
    """Half-band = b0 + slope * |gravity torque| on the loaded joint; constant where the load barely varies."""
    fits = {f.joint: f for f in fit_stiction(_session(LARGE_OFFSET_DEG), joint_map, raw_model, SIM, SagFitConfig())}
    lift = fits["shoulder_lift"]
    assert np.rad2deg(lift.half_band_rad) == pytest.approx(BAND0_DEG[1], abs=0.1)
    assert np.rad2deg(lift.load_slope_rad_per_nm) == pytest.approx(SLOPE_DEG_PER_NM, rel=0.1)
    assert np.rad2deg(fits["shoulder_pan"].half_band_rad) == pytest.approx(BAND0_DEG[0], abs=0.05)
    assert fits["shoulder_pan"].load_slope_rad_per_nm == 0.0


def test_small_approaches_give_no_stiction_fit(joint_map, raw_model):
    """A 3° approach cannot sweep the whole band, so the session is not used for friction."""
    assert fit_stiction(_session(SMALL_OFFSET_DEG), joint_map, raw_model, SIM, SagFitConfig()) == ()


def _calibration(joint_map, raw_model, offset_deg: float) -> ActuatorCalibration:
    """Calibration written from the synthetic session."""
    analysis = analyze_session(_session(offset_deg), joint_map, raw_model, SIM, SagFitConfig(), residual_max_rad=0.1)
    return calibration_from_analysis(analysis, source="unit test")


def test_calibration_stores_stiction_instead_of_mujoco_friction(joint_map, raw_model):
    """With a stiction fit, MuJoCo's creeping frictionloss is left off."""
    entries = {e.joint: e for e in _calibration(joint_map, raw_model, LARGE_OFFSET_DEG).joints}
    lift = entries["shoulder_lift"]
    assert lift.frictionloss_nm is None
    assert np.rad2deg(lift.stiction_half_band_rad) == pytest.approx(BAND0_DEG[1], abs=0.1)
    assert lift.stiction_load_slope_rad_per_nm > 0


def test_unloaded_joints_get_the_mean_fitted_stiffness_flagged_as_assumed(joint_map, raw_model):
    """Pan and wrist roll carry no gravity load: same servo, same gain, so assume the measured mean."""
    entries = {e.joint: e for e in _calibration(joint_map, raw_model, LARGE_OFFSET_DEG).joints}
    measured = [e.kp_nm_per_rad for e in entries.values() if not e.kp_assumed]
    assert entries["shoulder_pan"].kp_assumed and entries["wrist_roll"].kp_assumed
    assert entries["shoulder_pan"].kp_nm_per_rad == pytest.approx(np.mean(measured))
    assert not entries["shoulder_lift"].kp_assumed


def test_calibration_with_stiction_round_trips(joint_map, raw_model, tmp_path):
    """The new fields survive save/load."""
    calibration = _calibration(joint_map, raw_model, LARGE_OFFSET_DEG)
    calibration.save(tmp_path / "c.json")
    assert ActuatorCalibration.load(tmp_path / "c.json") == calibration


def test_headroom_warning_uses_the_worst_single_approach(joint_map, raw_model):
    """A joint whose mean sag is fine but whose against-gravity arrival is not still triggers the warning."""
    session = _session(LARGE_OFFSET_DEG)
    cap = 0.1
    limit_deg = np.rad2deg(SagFitConfig().early_warning_fraction * cap)
    goal = POSES_DEG[0]
    above = replace(session.samples[0], measured_deg=(*np.array(goal[:5]) - 0.3 * limit_deg, goal[5]))
    below = replace(session.samples[1], measured_deg=(*np.array(goal[:5]) - 1.5 * limit_deg, goal[5]))
    one_pose = replace(session, samples=(above, below))
    analysis = analyze_session(one_pose, joint_map, raw_model, SIM, SagFitConfig(), residual_max_rad=cap)
    assert set(analysis.headroom_warnings) == set(SIM.arm_joints)

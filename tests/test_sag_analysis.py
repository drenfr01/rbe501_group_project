"""Phase 0 analysis: per-joint summary, calibration file, sim comparison, and README text."""

from dataclasses import replace

import numpy as np
import pytest

from rbe_501_so101.config import HardwareConfig, JointMappingEntry, SagCheckConfig, SagFitConfig, SimConfig
from rbe_501_so101.kinematics.gravity import GravityTorqueModel
from rbe_501_so101.rl.plots import plot_real_sag
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.sag import SagSample, SagSession
from rbe_501_so101.hardware.sag_analysis import (
    analyze_session,
    calibration_from_analysis,
    compare_sag,
    sag_readme,
    simulate_session,
)

SIM = SimConfig()
IDENTITY_HW = HardwareConfig(joint_map=tuple(JointMappingEntry() for _ in SIM.arm_joints))
POSES_DEG = (
    (0.0, -20.3, 40.1, 15.2, 0.0, 40.0),
    (10.2, 0.4, 20.3, 40.1, 30.0, 40.0),
    (-15.1, 40.2, -20.4, -30.3, -40.0, 40.0),
    (5.3, 55.1, -50.2, 0.1, 20.0, 40.0),
)
TRUE_KP = np.array([20.0, 4.0, 20.0, 6.0, 4.0])
HYST_DEG = 0.3
HYSTERETIC_JOINTS = (1, 2)


def _sag_deg(goal: tuple[float, ...]) -> np.ndarray:
    """Spring-like sag `tau_g / kp` at a goal pose, in degrees (gripper: none)."""
    tau = GravityTorqueModel.from_config(SIM).torque(np.deg2rad(goal))
    return np.append(np.rad2deg(tau / TRUE_KP), 0.0)


def _session() -> SagSession:
    """Synthetic session: physical sag plus symmetric hysteresis on two joints."""
    samples = []
    for pid, goal in enumerate(POSES_DEG):
        sag = _sag_deg(goal)
        for approach, sign in (("above", 1.0), ("below", -1.0)):
            hyst = np.array([sign * HYST_DEG if i in HYSTERETIC_JOINTS else 0.0 for i in range(6)])
            measured = tuple((np.array(goal) - sag + hyst).tolist())
            samples.append(SagSample(pid, approach, goal, measured, (0.0,) * 6, ()))
    return SagSession("test", SIM.joints, (), tuple(samples))


@pytest.fixture
def analysis(raw_model):
    """Analysis of the synthetic session."""
    joint_map = JointMap.from_model(raw_model, SIM, IDENTITY_HW)
    return analyze_session(_session(), joint_map, raw_model, SIM, SagFitConfig(), residual_max_rad=0.1)


def test_summary_reports_sag_and_hysteresis_per_joint(analysis):
    joints = {j.joint: j for j in analysis.joints}
    expected = np.mean([_sag_deg(p)[1] for p in POSES_DEG])
    assert joints["shoulder_lift"].mean_sag_deg == pytest.approx(expected)
    assert joints["shoulder_lift"].fit.kp_nm_per_rad == pytest.approx(TRUE_KP[1], rel=0.05)
    assert joints["shoulder_lift"].mean_abs_hysteresis_deg == pytest.approx(HYST_DEG)
    assert joints["wrist_flex"].mean_abs_hysteresis_deg == pytest.approx(0.0)
    assert joints["shoulder_lift"].mjcf_kp == pytest.approx(20.0)
    assert analysis.n_poses == len(POSES_DEG)


def test_large_sag_triggers_the_headroom_warning(analysis):
    assert analysis.headroom_warnings == ("shoulder_lift",)


def test_calibration_only_includes_identifiable_joints(analysis):
    calibration = calibration_from_analysis(analysis, source="unit test")
    names = {e.joint for e in calibration.joints}
    assert "shoulder_pan" not in names
    lift = next(e for e in calibration.joints if e.joint == "shoulder_lift")
    assert lift.kp_nm_per_rad is not None and lift.frictionloss_nm is not None
    wrist = next((e for e in calibration.joints if e.joint == "wrist_flex"), None)
    assert wrist is None or wrist.frictionloss_nm is None


def test_readme_states_the_early_warning(analysis):
    text = sag_readme(analysis)
    assert "shoulder_lift" in text
    assert "Early warning" in text
    assert "±0.1 rad" in text


def test_real_sag_plot_is_written(analysis, tmp_path):
    path = tmp_path / "real_sag.png"
    plot_real_sag(analysis, path)
    assert path.stat().st_size > 0


def test_simulated_session_compares_joint_by_joint(raw_model):
    joint_map = JointMap.from_model(raw_model, SIM, IDENTITY_HW)
    real = _session()
    simulated = simulate_session(real, SIM, joint_map, SagCheckConfig(settle_s=1.0, n_reads=3), quantization_rad=0.0)
    rows = compare_sag(real, simulated, SIM.arm_joints)
    assert [r.joint for r in rows] == list(SIM.arm_joints)
    lift = next(r for r in rows if r.joint == "shoulder_lift")
    assert lift.real_mean_sag_deg == pytest.approx(np.mean([_sag_deg(p)[1] for p in POSES_DEG]))
    sim_expected = lift.real_mean_sag_deg * TRUE_KP[1] / 20.0
    assert lift.sim_mean_sag_deg == pytest.approx(sim_expected, rel=0.1)


def test_analysis_excludes_clamped_samples_and_reports_them(raw_model):
    joint_map = JointMap.from_model(raw_model, SIM, IDENTITY_HW)
    clean = _session()
    junk = tuple(replace(s, pose_id=99, measured_deg=(0.0,) * 6, clamped=True) for s in clean.samples[:2])
    dirty = replace(clean, samples=clean.samples + junk)
    expected = analyze_session(clean, joint_map, raw_model, SIM, SagFitConfig(), residual_max_rad=0.1)
    result = analyze_session(dirty, joint_map, raw_model, SIM, SagFitConfig(), residual_max_rad=0.1)
    assert result.n_excluded_clamped == 2 and result.n_poses == expected.n_poses
    assert result.joints == expected.joints
    assert "2 samples excluded" in sag_readme(result)

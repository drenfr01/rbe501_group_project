"""ActuatorCalibration: which model fields it changes, persistence, and backend wiring."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from rbe_501_so101.config import DynamicsRandomizationConfig, SimConfig
from rbe_501_so101.envs.calibration import ActuatorCalibration, JointActuatorCalibration
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend

SIM = SimConfig()
CALIBRATION = ActuatorCalibration(
    joints=(
        JointActuatorCalibration("shoulder_lift", kp_nm_per_rad=11.0, frictionloss_nm=0.05),
        JointActuatorCalibration("wrist_flex", kp_nm_per_rad=3.0),
    ),
    source="test",
)


def _array_snapshot(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Copy of every numeric array attribute of the model."""
    snapshot = {}
    for name in dir(model):
        value = getattr(model, name, None)
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
            snapshot[name] = value.copy()
    return snapshot


def _changed(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> set[str]:
    """Names of arrays whose contents differ."""
    return {k for k in before if not np.array_equal(before[k], after[k])}


def test_apply_changes_exactly_the_intended_fields(raw_model):
    before = _array_snapshot(raw_model)
    CALIBRATION.apply(raw_model, SIM)
    after = _array_snapshot(raw_model)
    assert _changed(before, after) == {"actuator_gainprm", "actuator_biasprm", "dof_frictionloss"}
    lift = raw_model.actuator("shoulder_lift_motor").id
    pan = raw_model.actuator("shoulder_pan_motor").id
    assert raw_model.actuator_gainprm[lift, 0] == pytest.approx(11.0)
    assert raw_model.actuator_biasprm[lift, 1] == pytest.approx(-11.0)
    np.testing.assert_array_equal(raw_model.actuator_gainprm[pan], before["actuator_gainprm"][pan])
    assert raw_model.dof_frictionloss[raw_model.joint("shoulder_lift").dofadr[0]] == pytest.approx(0.05)
    assert raw_model.dof_frictionloss[raw_model.joint("wrist_flex").dofadr[0]] == 0.0


def test_apply_preserves_damping_ratio(raw_model):
    lift = raw_model.actuator("shoulder_lift_motor").id
    kp0, kv0 = raw_model.actuator_gainprm[lift, 0], raw_model.actuator_biasprm[lift, 2]
    CALIBRATION.apply(raw_model, SIM)
    assert raw_model.actuator_biasprm[lift, 2] == pytest.approx(kv0 * np.sqrt(11.0 / kp0))


def test_round_trips_through_json(tmp_path):
    path = tmp_path / "cal.json"
    CALIBRATION.save(path)
    assert ActuatorCalibration.load(path) == CALIBRATION


def test_unknown_joint_is_rejected(raw_model):
    bad = ActuatorCalibration(joints=(JointActuatorCalibration("elbow", kp_nm_per_rad=1.0),))
    with pytest.raises(KeyError):
        bad.apply(raw_model, SIM)


def test_backend_without_calibration_matches_raw_model(backend, raw_model, rng):
    backend.reset(rng)
    np.testing.assert_array_equal(backend.model.actuator_gainprm, raw_model.actuator_gainprm)
    np.testing.assert_array_equal(backend.model.dof_frictionloss, raw_model.dof_frictionloss)


def test_backend_applies_calibration_before_nominal_capture(tmp_path, rng):
    path = tmp_path / "cal.json"
    CALIBRATION.save(path)
    randomization = DynamicsRandomizationConfig(
        enabled=True, kp_scale_range=(0.5, 0.5), damping_scale_range=(1.0, 1.0), payload_mass_range_kg=(0.0, 0.0)
    )
    backend = MujocoSo101Backend(replace(SIM, calibration_path=path), randomization)
    backend.reset(rng)
    lift = backend.model.actuator("shoulder_lift_motor").id
    assert backend.model.actuator_gainprm[lift, 0] == pytest.approx(5.5)
    backend.close()

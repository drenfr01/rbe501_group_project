"""Phase 0: sag data, repository, procedure order, and the stiffness/friction fits."""

from dataclasses import replace

import numpy as np
import pytest

from rbe_501_so101.config import (
    ENCODER_QUANTIZATION_RAD,
    DynamicsRandomizationConfig,
    HardwareConfig,
    SagCheckConfig,
    SagFitConfig,
    SimConfig,
)
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend
from rbe_501_so101.hardware.bus import FakeJointBus
from rbe_501_so101.hardware.clock import FakeClock
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.sag import (
    RecordedPose,
    SagRepository,
    SagSample,
    SagSession,
    fit_compliance,
    friction_torque,
    headroom_warnings,
    hysteresis_by_joint,
    sag_by_joint,
    usable_samples,
)
from rbe_501_so101.hardware.sag_procedure import SagCheckRunner
from rbe_501_so101.hardware.sim_bus import SimJointBus
from rbe_501_so101.hardware.types import MotorStatus
from rbe_501_so101.stats import student_t_ppf

SIM = SimConfig()
JOINTS = SIM.joints
FAST_SAG_CFG = SagCheckConfig(settle_s=1.0, n_reads=5, ramp_speed_deg_s=120.0)
KP_SCALE = 0.6
FRICTION_NM = 0.02
JOG_JITTER_SEED = 0
FIT_POSES_DEG = (
    (0.0, -20.0, 40.0, 15.0, 0.0, 40.0),
    (10.0, 0.0, 20.0, 40.0, 30.0, 40.0),
    (-15.0, 40.0, -20.0, -30.0, -40.0, 40.0),
    (20.0, -40.0, 70.0, 10.0, 60.0, 40.0),
    (-25.0, 0.0, -40.0, 50.0, -10.0, 40.0),
    (5.0, 55.0, -50.0, 0.0, 20.0, 40.0),
    (0.0, -60.0, 80.0, -40.0, 0.0, 40.0),
    (-10.0, 30.0, 30.0, -60.0, 50.0, 40.0),
)


def _sample(pose_id: int, approach: str, goal: list[float], measured: list[float]) -> SagSample:
    """Hand-built sample with zero spread and nominal status."""
    status = (MotorStatus("shoulder_pan", 30.0, 12.0, 0.1),)
    return SagSample(pose_id, approach, tuple(goal), tuple(measured), tuple(0.0 for _ in goal), status)


def _session(samples: tuple[SagSample, ...]) -> SagSession:
    """Session over the sim joint names with no recorded settings."""
    return SagSession(robot_id="test", joints=JOINTS, settings=(), samples=samples)


def _hand_session() -> SagSession:
    """Pose 0: sag 2 deg (above) and 1 deg (below) on every joint."""
    goal = [10.0] * 6
    return _session(
        (
            _sample(0, "above", goal, [8.0] * 6),
            _sample(0, "below", goal, [9.0] * 6),
        )
    )


def test_student_t_quantile_matches_tables():
    assert student_t_ppf(0.975, 10) == pytest.approx(2.228, abs=1e-3)
    assert student_t_ppf(0.975, 1000) == pytest.approx(1.962, abs=1e-3)


def test_repository_round_trips_session(tmp_path):
    repo = SagRepository(SagCheckConfig(sessions_dir=tmp_path))
    session = _hand_session()
    path = repo.save_session(session, repo.session_dir(session.robot_id))
    assert repo.load_session(path) == session


def test_repository_round_trips_poses(tmp_path):
    cfg = SagCheckConfig(poses_path=tmp_path / "poses" / "poses.json")
    repo = SagRepository(cfg)
    poses = (RecordedPose(0, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),)
    repo.save_poses(JOINTS, poses)
    assert repo.load_poses() == (JOINTS, poses)


def test_sag_and_hysteresis_on_hand_built_samples():
    session = _hand_session()
    table = sag_by_joint(session)
    np.testing.assert_allclose(table.values_deg, [[1.5] * 6])
    assert table.pose_ids == (0,)
    # Arriving from above settles 1 deg lower than from below: measured_above - measured_below = -1.
    np.testing.assert_allclose(hysteresis_by_joint(session).values_deg, [[-0.5] * 6])


def test_friction_torque_is_stiffness_times_hysteresis():
    np.testing.assert_allclose(friction_torque(np.array([0.01, -0.02]), np.array([10.0, 5.0])), [0.1, 0.1])


def test_headroom_warning_flags_large_sag():
    session = _session((_sample(0, "above", [10.0] * 6, [10.0, 6.0, 10.0, 10.0, 10.0, 10.0]),))
    identity = JointMap(SIM.arm_joints, SIM.gripper_joint, HardwareConfig().joint_map, (0.0, 100.0), (0.0, 1.0))
    warnings = headroom_warnings(sag_by_joint(session), identity, 0.1, 0.5)
    assert warnings == ("shoulder_lift",)


def test_runner_approaches_each_pose_from_both_sides_in_order():
    bus = FakeJointBus(JOINTS, sag_deg={"shoulder_lift": 1.0})
    poses = (RecordedPose(0, (0.0,) * 6), RecordedPose(1, (10.0, 20.0, 30.0, 0.0, 0.0, 50.0)))
    runner = SagCheckRunner(bus, FakeClock(), FAST_SAG_CFG, JOINTS)
    samples = runner.run(poses)
    assert [(s.pose_id, s.approach) for s in samples] == [(0, "above"), (0, "below"), (1, "above"), (1, "below")]
    assert samples[0].measured_deg[1] == pytest.approx(-1.0)
    offset = FAST_SAG_CFG.approach_offset_deg
    pose_goals = [i for i, g in enumerate(bus.sent) if g == runner.goal_dict(poses[1].positions_deg)]
    before_above, before_below = bus.sent[pose_goals[0] - 1], bus.sent[pose_goals[1] - 1]
    assert before_above[pos_key("elbow_flex")] == pytest.approx(30.0 + offset)
    assert before_below[pos_key("elbow_flex")] == pytest.approx(30.0 - offset)
    assert before_above[pos_key("gripper")] == pytest.approx(50.0)


def test_runner_finish_ramps_to_rest_then_disables_torque():
    bus = FakeJointBus(JOINTS)
    runner = SagCheckRunner(bus, FakeClock(), FAST_SAG_CFG, JOINTS)
    rest = {pos_key(j): 5.0 for j in JOINTS}
    runner.finish(rest)
    assert bus.sent[-1] == rest
    assert bus.torque_log[-1] is False


def _jogged(poses: tuple[tuple[float, ...], ...]) -> list[tuple[float, ...]]:
    """Add sub-degree jitter to the arm joints, like hand-jogged poses.

    Whole-degree poses with similar sag all round the same way on the encoder
    grid, which biases the fit without showing up in its residuals.
    """
    jitter = np.random.default_rng(JOG_JITTER_SEED).uniform(-0.5, 0.5, size=(len(poses), len(JOINTS)))
    jitter[:, -1] = 0.0
    return [tuple((np.array(p) + j).tolist()) for p, j in zip(poses, jitter, strict=True)]


@pytest.fixture(scope="module")
def imperfect_arm_session() -> SagSession:
    """Run the Phase 0 procedure on a softer, frictional, rounded MuJoCo arm."""
    randomization = DynamicsRandomizationConfig(
        enabled=True, kp_scale_range=(KP_SCALE, KP_SCALE), damping_scale_range=(1.0, 1.0), payload_mass_range_kg=(0.0, 0.0)
    )
    backend = MujocoSo101Backend(SIM, randomization)
    joint_map = JointMap.from_model(backend.model, SIM, HardwareConfig())
    bus = SimJointBus(backend, joint_map, SIM, quantization_rad=ENCODER_QUANTIZATION_RAD)
    backend.model.dof_frictionloss[backend.ids.arm_dof_adr] = FRICTION_NM
    poses = tuple(RecordedPose(i, p) for i, p in enumerate(_jogged(FIT_POSES_DEG)))
    runner = SagCheckRunner(bus, bus, FAST_SAG_CFG, JOINTS)
    session = runner.session("sim", poses)
    bus.close()
    return session


def test_fit_recovers_known_stiffness_from_imperfect_arm(imperfect_arm_session, raw_model):
    joint_map = JointMap.from_model(raw_model, SIM, HardwareConfig())
    fits = {f.joint: f for f in fit_compliance(imperfect_arm_session, joint_map, raw_model, SIM, SagFitConfig())}
    nominal_kp = raw_model.actuator_gainprm[raw_model.actuator("shoulder_lift_motor").id, 0]
    for joint in ("shoulder_lift", "elbow_flex"):
        fit = fits[joint]
        true_kp = KP_SCALE * nominal_kp
        assert fit.identifiable
        assert fit.kp_ci_low <= true_kp <= fit.kp_ci_high, (joint, fit)


def test_joint_with_constant_gravity_torque_is_not_identifiable(imperfect_arm_session, raw_model):
    joint_map = JointMap.from_model(raw_model, SIM, HardwareConfig())
    fits = {f.joint: f for f in fit_compliance(imperfect_arm_session, joint_map, raw_model, SIM, SagFitConfig())}
    assert not fits["shoulder_pan"].identifiable
    assert fits["shoulder_pan"].kp_nm_per_rad is None


def test_wide_span_requirement_marks_everything_unidentifiable(imperfect_arm_session, raw_model):
    joint_map = JointMap.from_model(raw_model, SIM, HardwareConfig())
    cfg = replace(SagFitConfig(), min_torque_span_nm=1e3)
    assert not any(f.identifiable for f in fit_compliance(imperfect_arm_session, joint_map, raw_model, SIM, cfg))


def test_fake_bus_clips_goals_relative_to_present_like_lerobot():
    bus = FakeJointBus(JOINTS, max_relative_deg=10.0)
    sent = bus.send_positions({pos_key("elbow_flex"): 25.0, pos_key("shoulder_pan"): -4.0})
    assert sent == {pos_key("elbow_flex"): 10.0, pos_key("shoulder_pan"): -4.0}
    assert bus.read_positions()[pos_key("elbow_flex")] == pytest.approx(10.0)


def test_runner_records_sent_goal_and_leaves_unclamped_samples_unflagged():
    bus = FakeJointBus(JOINTS, sag_deg={"shoulder_lift": 1.0}, max_relative_deg=10.0)
    samples = SagCheckRunner(bus, FakeClock(), FAST_SAG_CFG, JOINTS).run((RecordedPose(0, (0.0, 20.0, 30.0, 0.0, 0.0, 50.0)),))
    assert [s.clamped for s in samples] == [False, False]
    assert samples[0].sent_goal_deg == samples[0].goal_deg


def test_runner_flags_samples_where_lerobot_clamped_a_goal():
    bus = FakeJointBus(JOINTS, sag_deg={"shoulder_lift": 12.0}, max_relative_deg=10.0)
    samples = SagCheckRunner(bus, FakeClock(), FAST_SAG_CFG, JOINTS).run((RecordedPose(0, (0.0, 20.0, 30.0, 0.0, 0.0, 50.0)),))
    assert all(s.clamped for s in samples)
    assert samples[0].sent_goal_deg[1] != pytest.approx(samples[0].goal_deg[1])


def test_session_round_trips_clamp_fields_and_loads_older_sessions(tmp_path):
    flagged = replace(_hand_session().samples[0], sent_goal_deg=(9.0,) * 6, clamped=True)
    session = _session((flagged,))
    assert SagSession.from_dict(session.to_dict()) == session
    old = session.to_dict()
    for sample in old["samples"]:
        del sample["sent_goal_deg"], sample["clamped"]
    loaded = SagSession.from_dict(old).samples[0]
    assert loaded.clamped is None and loaded.sent_goal_deg is None


def test_usable_samples_drops_only_clamped_ones():
    clean, unknown = _hand_session().samples
    session = _session((clean, replace(unknown, clamped=None), replace(clean, pose_id=1, clamped=True)))
    kept = usable_samples(session)
    assert [(s.pose_id, s.approach) for s in kept.samples] == [(0, "above"), (0, "below")]

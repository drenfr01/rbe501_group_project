"""SafetyMonitor and LeRobotSo101Backend against FakeJointBus and a fake clock."""

from dataclasses import replace

import numpy as np
import pytest

from rbe_501_so101.config import HardwareConfig, IKConfig, JointMappingEntry, ReachTaskConfig, SafetyConfig, SimConfig
from rbe_501_so101.envs.protocols import RobotBackend
from rbe_501_so101.envs.reach_env import SO101ReachEnv
from rbe_501_so101.envs.targets import SequenceTargetSampler
from rbe_501_so101.envs.types import ObservationLayout
from rbe_501_so101.hardware.bus import FakeJointBus
from rbe_501_so101.hardware.clock import FakeClock
from rbe_501_so101.hardware.joint_mapping import JointMap
from rbe_501_so101.hardware.real_backend import LeRobotSo101Backend
from rbe_501_so101.hardware.safety import SafetyAbort, SafetyMonitor
from rbe_501_so101.hardware.types import MotorStatus
from rbe_501_so101.kinematics import PositionIKSolver
from rbe_501_so101.rl.evaluation import run_episode
from rbe_501_so101.rl.policies import IKHoldPolicy

SIM = SimConfig()
HW = HardwareConfig(joint_map=tuple(JointMappingEntry() for _ in range(5)))
N_ARM = len(SIM.arm_joints)
HOME_DEG = (0.0, -20.0, 40.0, 15.0, 0.0, 40.0)
BELOW_TABLE_DEG = (0.0, 90.0, 0.0, 0.0, 0.0, 40.0)
START_DEG = {"shoulder_pan": 10.0, "shoulder_lift": -40.0, "elbow_flex": 60.0, "wrist_flex": 0.0, "wrist_roll": 5.0}
TARGET = np.array([0.15, 0.0, 0.6])


class SlowBus(FakeJointBus):
    """Fake bus whose every send takes `delay_s` of (fake) time."""

    def __init__(self, clock: FakeClock, delay_s: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._clock = clock
        self._delay_s = delay_s

    def send_positions(self, goal: dict[str, float]) -> dict[str, float]:
        """Advance the clock, then behave like the fake bus."""
        self._clock.advance(self._delay_s)
        return super().send_positions(goal)


@pytest.fixture
def kinematics() -> PositionIKSolver:
    """Forward kinematics on a private model."""
    return PositionIKSolver.from_config(SIM, IKConfig())


@pytest.fixture
def safety(raw_model) -> SafetyMonitor:
    """Safety monitor with the default thresholds."""
    return SafetyMonitor.from_model(raw_model, SIM, SafetyConfig())


def _backend(bus, clock, safety, kinematics, raw_model, hw=HW) -> LeRobotSo101Backend:
    """Real backend wired to fakes."""
    joint_map = JointMap.from_model(raw_model, SIM, hw)
    return LeRobotSo101Backend(bus, joint_map, safety, kinematics, raw_model, SIM, hw, clock, HOME_DEG)


def _bus(**kwargs) -> FakeJointBus:
    """Fake bus starting away from home."""
    return FakeJointBus(SIM.joints, start_deg=START_DEG, **kwargs)


def test_backend_implements_protocol(safety, kinematics, raw_model):
    backend = _backend(_bus(), FakeClock(), safety, kinematics, raw_model)
    assert isinstance(backend, RobotBackend)


def test_reset_holds_present_then_ramps_monotonically_home(safety, kinematics, raw_model):
    bus = _bus()
    backend = _backend(bus, FakeClock(), safety, kinematics, raw_model)
    present = bus.read_positions()
    state = backend.reset(np.random.default_rng(0))
    assert bus.torque_log == [True]
    assert bus.sent[0] == pytest.approx(present)
    ramp = np.array([[g[f"{j}.pos"] for j in SIM.arm_joints] for g in bus.sent])
    np.testing.assert_allclose(ramp[-1], HOME_DEG[:N_ARM])
    np.testing.assert_allclose(state.q_cmd[:N_ARM], np.deg2rad(HOME_DEG[:N_ARM]))


def _per_joint_monotonic(ramp: np.ndarray) -> bool:
    """Each column never changes direction."""
    steps = np.diff(ramp, axis=0)
    return all(np.all(col >= -1e-9) or np.all(col <= 1e-9) for col in steps.T)


def test_reset_ramp_is_monotonic_per_joint(safety, kinematics, raw_model):
    bus = _bus()
    _backend(bus, FakeClock(), safety, kinematics, raw_model).reset(np.random.default_rng(0))
    ramp = np.array([[g[f"{j}.pos"] for j in SIM.arm_joints] for g in bus.sent])
    assert _per_joint_monotonic(ramp)
    assert len(bus.sent) > 2


def test_step_sends_mapped_command_and_waits_out_the_period(safety, kinematics, raw_model):
    bus, clock = _bus(), FakeClock()
    backend = _backend(bus, clock, safety, kinematics, raw_model)
    backend.reset(np.random.default_rng(0))
    backend.apply_delta(np.full(N_ARM, 0.01))
    start = clock.now
    state = backend.step()
    expected = JointMap.from_model(raw_model, SIM, HW).to_lerobot(state.q_cmd)
    assert bus.sent[-1] == pytest.approx(expected)
    assert clock.now - start == pytest.approx(SIM.control_dt)
    assert backend.overrun_count == 0


def test_step_reports_differenced_velocity_and_fk_position(safety, kinematics, raw_model):
    bus = _bus(sag_deg={"shoulder_lift": 1.0})
    backend = _backend(bus, FakeClock(), safety, kinematics, raw_model)
    first = backend.reset(np.random.default_rng(0))
    backend.apply_delta(np.full(N_ARM, 0.02))
    second = backend.step()
    np.testing.assert_allclose(second.qvel, (second.qpos - first.qpos) / SIM.control_dt)
    np.testing.assert_allclose(second.tcp_pos, kinematics.forward(second.qpos))
    assert second.q_cmd[1] - second.qpos[1] == pytest.approx(np.deg2rad(1.0))


def test_apply_delta_clips_to_tighter_limits_minus_margin(safety, kinematics, raw_model):
    real_limits = {j: (-30.0, 30.0) for j in SIM.joints}
    bus = _bus(limits=real_limits)
    backend = _backend(bus, FakeClock(), safety, kinematics, raw_model)
    limits = backend.joint_limits()
    np.testing.assert_allclose(limits.upper, np.deg2rad(30.0) - HW.limit_margin_rad)
    np.testing.assert_allclose(limits.lower, -np.deg2rad(30.0) + HW.limit_margin_rad)
    wide = _backend(_bus(), FakeClock(), safety, kinematics, raw_model).joint_limits()
    sim_upper = raw_model.jnt_range[[raw_model.joint(j).id for j in SIM.arm_joints], 1]
    np.testing.assert_allclose(wide.upper, sim_upper - HW.limit_margin_rad)


def test_close_ramps_to_rest_before_disabling_torque(safety, kinematics, raw_model):
    bus = _bus()
    backend = _backend(bus, FakeClock(), safety, kinematics, raw_model)
    backend.reset(np.random.default_rng(0))
    n_before = len(bus.sent)
    backend.close()
    rest = JointMap.from_model(raw_model, SIM, HW).to_lerobot(np.deg2rad(HW.rest_deg))
    assert len(bus.sent) > n_before
    assert bus.sent[-1] == pytest.approx(rest)
    assert bus.torque_log[-1] is False
    assert bus.closed


def test_safety_rejects_pose_below_table(safety):
    safety.check_command(np.deg2rad(HOME_DEG))
    with pytest.raises(SafetyAbort, match="table"):
        safety.check_command(np.deg2rad(BELOW_TABLE_DEG))


def test_safety_aborts_on_temperature_and_load(safety):
    safety.check_status((MotorStatus("elbow_flex", 40.0, 12.0, 0.3),))
    with pytest.raises(SafetyAbort, match="temperature"):
        safety.check_status((MotorStatus("elbow_flex", 70.0, 12.0, 0.3),))
    with pytest.raises(SafetyAbort, match="load"):
        safety.check_status((MotorStatus("elbow_flex", 40.0, 12.0, 0.95),))


def test_safety_aborts_after_consecutive_overruns(safety):
    limit = SafetyConfig().max_consecutive_overruns
    for _ in range(limit - 1):
        safety.record_timing(overran=True)
    safety.record_timing(overran=False)
    for _ in range(limit - 1):
        safety.record_timing(overran=True)
    with pytest.raises(SafetyAbort, match="overrun"):
        safety.record_timing(overran=True)


def test_repeated_overruns_abort_and_ramp_to_rest(safety, kinematics, raw_model):
    clock = FakeClock()
    bus = SlowBus(clock, 2 * SIM.control_dt, joints=SIM.joints, start_deg=START_DEG)
    hw = replace(HW, home_ramp_s=SIM.control_dt)
    backend = _backend(bus, clock, safety, kinematics, raw_model, hw)
    backend.reset(np.random.default_rng(0))
    with pytest.raises(SafetyAbort):
        for _ in range(SafetyConfig().max_consecutive_overruns + 1):
            backend.step()
    assert backend.overrun_count >= SafetyConfig().max_consecutive_overruns
    assert bus.torque_log[-1] is False


def test_unsafe_command_is_never_sent(safety, kinematics, raw_model):
    bus = _bus()
    backend = _backend(bus, FakeClock(), safety, kinematics, raw_model)
    backend.reset(np.random.default_rng(0))
    backend.apply_delta(np.deg2rad(np.array(BELOW_TABLE_DEG[:N_ARM]) - np.array(HOME_DEG[:N_ARM])))
    joint_map = JointMap.from_model(raw_model, SIM, HW)
    with pytest.raises(SafetyAbort):
        backend.step()
    unsafe = joint_map.to_lerobot(np.deg2rad(BELOW_TABLE_DEG))
    assert all(g != pytest.approx(unsafe) for g in bus.sent)


def test_reach_env_rollout_with_ik_hold_on_fake_arm(safety, kinematics, raw_model):
    bus = _bus(sag_deg={"shoulder_lift": 1.0, "elbow_flex": 0.5})
    backend = _backend(bus, FakeClock(), safety, kinematics, raw_model)
    env = SO101ReachEnv(backend=backend, sampler=SequenceTargetSampler((TARGET,)))
    policy = IKHoldPolicy(kinematics, ReachTaskConfig(), ObservationLayout(n_arm=N_ARM))
    stats = run_episode(env, policy, seed=0, window=10)
    assert stats is not None and np.all(np.isfinite(stats.holding_errors_m))
    env.close()
    assert bus.torque_log[-1] is False

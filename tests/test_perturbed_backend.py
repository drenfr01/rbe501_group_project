"""PerturbedBackend (delay, encoder rounding) and the shared reach-env factory."""

from dataclasses import replace

import numpy as np
import pytest

from rbe_501_so101.config import (
    ENCODER_QUANTIZATION_RAD,
    PRESETS,
    EnvOptions,
    IKConfig,
    PerturbationConfig,
    SimConfig,
)
from rbe_501_so101.envs.calibration import ActuatorCalibration, JointActuatorCalibration
from rbe_501_so101.envs.factory import make_reach_env
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend
from rbe_501_so101.envs.perturbed_backend import PerturbedBackend
from rbe_501_so101.envs.protocols import RobotBackend, TargetMarker
from rbe_501_so101.kinematics import PositionIKSolver

SIM = SimConfig()
N_ARM = len(SIM.arm_joints)
DELTA = 0.05


@pytest.fixture
def solver() -> PositionIKSolver:
    """Forward kinematics on a private model copy."""
    return PositionIKSolver.from_config(SIM, IKConfig())


def _wrap(cfg: PerturbationConfig, solver: PositionIKSolver) -> PerturbedBackend:
    """Perturbed wrapper around a fresh MuJoCo backend."""
    return PerturbedBackend(MujocoSo101Backend(SIM), cfg, SIM.control_dt, solver)


def test_wrapper_is_a_robot_backend_with_marker(solver):
    backend = _wrap(PerturbationConfig(), solver)
    assert isinstance(backend, RobotBackend)
    assert isinstance(backend, TargetMarker)
    backend.close()


def test_delay_applies_delta_exactly_two_steps_later(solver, rng):
    backend = _wrap(PerturbationConfig(delay_steps=2), solver)
    inner = backend.inner
    home = backend.reset(rng).q_cmd.copy()
    backend.apply_delta(np.full(N_ARM, DELTA))
    commands = []
    for _ in range(3):
        backend.step()
        commands.append(inner.command[:N_ARM] - home[:N_ARM])
        backend.apply_delta(np.zeros(N_ARM))
    np.testing.assert_allclose(commands[0], 0.0)
    np.testing.assert_allclose(commands[1], 0.0)
    np.testing.assert_allclose(commands[2], DELTA)
    backend.close()


def test_delayed_backend_reports_the_intended_command(solver, rng):
    backend = _wrap(PerturbationConfig(delay_steps=1), solver)
    home = backend.reset(rng).q_cmd.copy()
    backend.apply_delta(np.full(N_ARM, DELTA))
    state = backend.step()
    np.testing.assert_allclose(state.q_cmd[:N_ARM] - home[:N_ARM], DELTA)
    backend.close()


def test_rounding_snaps_readings_and_differences_velocity(solver, rng):
    q = ENCODER_QUANTIZATION_RAD
    backend = _wrap(PerturbationConfig(quantization_rad=q), solver)
    first = backend.reset(rng)
    backend.apply_delta(np.full(N_ARM, DELTA))
    second = backend.step()
    np.testing.assert_allclose(second.qpos / q, np.round(second.qpos / q), atol=1e-9)
    np.testing.assert_allclose(second.qvel, (second.qpos - first.qpos) / SIM.control_dt)
    backend.close()


def test_rounded_tcp_matches_forward_kinematics(solver, rng):
    backend = _wrap(PerturbationConfig(quantization_rad=ENCODER_QUANTIZATION_RAD), solver)
    backend.reset(rng)
    backend.apply_delta(np.full(N_ARM, DELTA))
    state = backend.step()
    np.testing.assert_allclose(state.tcp_pos, solver.forward(state.qpos))
    backend.close()


def test_no_perturbation_passes_state_through(solver, rng):
    wrapped = _wrap(PerturbationConfig(), solver)
    plain = MujocoSo101Backend(SIM)
    for backend in (wrapped, plain):
        backend.reset(np.random.default_rng(0))
        backend.apply_delta(np.full(N_ARM, DELTA))
    a, b = wrapped.step(), plain.step()
    np.testing.assert_allclose(a.qpos, b.qpos)
    np.testing.assert_allclose(a.tcp_pos, b.tcp_pos)
    wrapped.close()
    plain.close()


def test_randomized_delay_stays_within_bounds(solver):
    backend = _wrap(PerturbationConfig(delay_steps=1, randomize_delay=True), solver)
    gen = np.random.default_rng(0)
    delays = set()
    for _ in range(20):
        backend.reset(gen)
        delays.add(backend.delay_steps)
    assert delays == {0, 1}
    backend.close()


def test_factory_default_builds_plain_mujoco_env():
    env = make_reach_env(PRESETS["residual_ik"], randomize=False)
    assert isinstance(env.backend, MujocoSo101Backend)
    env.close()


def test_factory_wraps_backend_when_perturbed():
    options = EnvOptions(perturbation=PerturbationConfig(delay_steps=1))
    env = make_reach_env(PRESETS["residual_ik"], randomize=False, options=options)
    assert isinstance(env.backend, PerturbedBackend)
    obs, _ = env.reset(seed=0)
    env.step(np.zeros(N_ARM, dtype=np.float32))
    assert np.all(np.isfinite(obs))
    env.close()


def test_factory_applies_calibration(tmp_path):
    path = tmp_path / "cal.json"
    ActuatorCalibration((JointActuatorCalibration("elbow_flex", kp_nm_per_rad=7.0),)).save(path)
    env = make_reach_env(PRESETS["legacy"], randomize=False, options=EnvOptions(calibration_path=str(path)))
    model = env.backend.model
    assert model.actuator_gainprm[model.actuator("elbow_flex_motor").id, 0] == pytest.approx(7.0)
    env.close()


def test_preset_env_options_round_trip_through_dict():
    preset = PRESETS["residual_ik_calibrated"]
    restored = type(preset).from_dict(preset.to_dict())
    assert restored == preset
    legacy_dict = {k: v for k, v in PRESETS["legacy"].to_dict().items() if k != "env"}
    assert type(preset).from_dict(legacy_dict).env == EnvOptions()
    assert replace(preset.env, calibration_path=None).perturbation.active

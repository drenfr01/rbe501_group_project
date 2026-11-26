"""ServoEffectsBackend: stiction as play on the servo goal, LeRobot's relative clip, and env wiring."""

import json

import numpy as np
import pytest

from rbe_501_so101.config import (
    PRESETS,
    EnvOptions,
    ExperimentPreset,
    PerturbationConfig,
    ServoEffectsConfig,
    SimConfig,
)
from rbe_501_so101.envs.calibration import ActuatorCalibration, JointActuatorCalibration
from rbe_501_so101.envs.factory import make_reach_env
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend
from rbe_501_so101.envs.perturbed_backend import PerturbedBackend
from rbe_501_so101.envs.protocols import RobotBackend
from rbe_501_so101.envs.servo_effects import ServoEffectsBackend, StictionParams, hysteresis, play
from rbe_501_so101.envs.types import JointLimits, RobotState

SIM = SimConfig()
N_ARM = len(SIM.arm_joints)
BAND = 0.02
LIMIT = 2.0
PLAY = ServoEffectsConfig(stiction=True, stiction_transition_ratio=0.0)


class IdealArm:
    """Fake backend whose joints sit exactly at the command it was given (or stay stuck)."""

    def __init__(self, stuck: bool = False) -> None:
        self.cmd = np.zeros(N_ARM + 1)
        self.stuck = stuck

    def reset(self, rng: np.random.Generator) -> RobotState:
        """Home at zero."""
        self.cmd = np.zeros(N_ARM + 1)
        return self._state()

    def apply_delta(self, delta_arm_rad: np.ndarray) -> None:
        """Integrate the arm command."""
        self.cmd[:N_ARM] += delta_arm_rad

    def step(self) -> RobotState:
        """The joint is wherever it was told to be (or at zero if stuck)."""
        return self._state()

    def joint_limits(self) -> JointLimits:
        """Wide symmetric limits."""
        return JointLimits(-LIMIT * np.ones(N_ARM), LIMIT * np.ones(N_ARM))

    def close(self) -> None:
        """Nothing to release."""

    def _state(self) -> RobotState:
        q = np.zeros_like(self.cmd) if self.stuck else self.cmd.copy()
        return RobotState(qpos=q, qvel=np.zeros_like(q), q_cmd=self.cmd.copy(), tcp_pos=np.zeros(3))


class ConstantTorque:
    """Gravity stub: the same holding torque everywhere."""

    def __init__(self, tau: float) -> None:
        self.tau = tau

    def torque(self, q: np.ndarray) -> np.ndarray:
        """Constant torque on every arm joint."""
        return np.full(N_ARM, self.tau)


def _params(band: float = BAND, slope: float = 0.0) -> StictionParams:
    """Same band on every arm joint."""
    return StictionParams(np.full(N_ARM, band), np.full(N_ARM, slope))


def _move(backend: ServoEffectsBackend, delta: float) -> RobotState:
    """Step the issued command by `delta` on every joint."""
    backend.apply_delta(np.full(N_ARM, delta))
    return backend.step()


def test_play_follows_only_outside_the_band():
    """The goal stays put while the command moves inside the band, then trails by the band."""
    goal = np.zeros(1)
    assert play(goal, np.array([0.5 * BAND]), np.array([BAND]))[0] == 0.0
    assert play(goal, np.array([3 * BAND]), np.array([BAND]))[0] == pytest.approx(2 * BAND)



def _sweep(ratio: float, targets: np.ndarray, band: float = BAND) -> np.ndarray:
    """Servo goal after each target of a one-joint command path, starting at rest at the first target."""
    goal, prev, out = targets[:1].copy(), targets[:1].copy(), []
    for t in targets[1:]:
        goal = hysteresis(goal, prev, np.array([t]), np.array([band]), ratio)
        prev = np.array([t])
        out.append(goal[0])
    return np.array(out)


def _zigzag(rng: np.random.Generator) -> np.ndarray:
    """A random command path with reversals of every size."""
    return np.cumsum(np.concatenate([[0.0], rng.normal(0.0, 0.6 * BAND, 400)]))


def test_hysteresis_with_zero_ratio_is_play(rng):
    """Ratio 0 is the sharp-cornered backlash operator."""
    targets = _zigzag(rng)
    goal, expected = targets[:1].copy(), []
    for t in targets[1:]:
        goal = play(goal, np.array([t]), np.array([BAND]))
        expected.append(goal[0])
    np.testing.assert_allclose(_sweep(0.0, targets), expected, atol=1e-12)


@pytest.mark.parametrize("ratio", [0.5, 1.5, 3.0])
def test_hysteresis_lag_stays_in_the_band_and_never_runs_backwards(rng, ratio):
    """The goal trails the command by at most the band and only moves the way the command moves."""
    targets = _zigzag(rng)
    goals = _sweep(ratio, targets)
    assert np.all(np.abs(targets[1:] - goals) <= BAND + 1e-12)
    moves, goal_moves = np.diff(targets[1:]), np.diff(goals)
    assert np.all(goal_moves * moves >= -1e-12)


def test_long_moves_saturate_at_the_full_band():
    """Far from any reversal the goal trails by the whole band, whatever the ratio."""
    targets = np.linspace(0.0, 30 * BAND, 300)
    assert targets[-1] - _sweep(1.5, targets)[-1] == pytest.approx(BAND, rel=1e-3)


def test_smooth_hysteresis_moves_on_small_reversals_where_play_sticks():
    """A reversal smaller than two bands leaves play's goal put; a smooth transition already slips."""
    up = np.linspace(0.0, 10 * BAND, 100)
    back = np.linspace(up[-1], up[-1] - 1.5 * BAND, 30)[1:]
    targets = np.concatenate([up, back])
    play_goals, smooth_goals = _sweep(0.0, targets), _sweep(1.5, targets)
    assert play_goals[-1] == pytest.approx(play_goals[len(up) - 2])
    slipped = smooth_goals[len(up) - 2] - smooth_goals[-1]
    assert 0.0 < slipped < 1.5 * BAND


def test_stiction_makes_the_arrival_side_decide_the_position(rng):
    """Arriving from below stops one band short; reversing by less than two bands does not move it."""
    inner = IdealArm()
    backend = ServoEffectsBackend(inner, _params(), PLAY)
    backend.reset(rng)
    state = _move(backend, 5 * BAND)
    np.testing.assert_allclose(state.qpos[:N_ARM], 4 * BAND)
    np.testing.assert_allclose(state.q_cmd[:N_ARM], 5 * BAND)
    state = _move(backend, -1.5 * BAND)
    np.testing.assert_allclose(state.qpos[:N_ARM], 4 * BAND)
    state = _move(backend, -1.5 * BAND)
    np.testing.assert_allclose(state.qpos[:N_ARM], 3 * BAND)
    assert isinstance(backend, RobotBackend)


def test_band_grows_with_gravity_load(rng):
    """Half-band = b0 + slope * |torque|."""
    slope, tau = 0.01, 2.0
    backend = ServoEffectsBackend(IdealArm(), _params(BAND, slope), PLAY, ConstantTorque(tau))
    backend.reset(rng)
    state = _move(backend, 0.2)
    np.testing.assert_allclose(state.qpos[:N_ARM], 0.2 - (BAND + slope * tau))


def test_stiction_scale_is_drawn_per_episode(rng):
    """A (2, 2) scale range doubles the band."""
    cfg = ServoEffectsConfig(stiction=True, stiction_scale_range=(2.0, 2.0), stiction_transition_ratio=0.0)
    backend = ServoEffectsBackend(IdealArm(), _params(), cfg)
    backend.reset(rng)
    np.testing.assert_allclose(_move(backend, 0.2).qpos[:N_ARM], 0.2 - 2 * BAND)


def test_relative_clip_limits_the_goal_around_the_measured_position(rng):
    """Like LeRobot's max_relative_target: a stuck joint never gets a goal further than the clip."""
    inner = IdealArm(stuck=True)
    clip = 0.1
    backend = ServoEffectsBackend(inner, _params(0.0), ServoEffectsConfig(relative_clip_rad=clip))
    backend.reset(rng)
    _move(backend, 0.5)
    np.testing.assert_allclose(inner.cmd[:N_ARM], clip)
    assert backend.command[0] == pytest.approx(0.5)


def _calibration_file(tmp_path, band_rad: float) -> str:
    """Calibration with the measured stiffness and a stiction band on every arm joint."""
    entries = tuple(
        JointActuatorCalibration(j, kp_nm_per_rad=11.0, stiction_half_band_rad=band_rad, stiction_load_slope_rad_per_nm=0.0)
        for j in SIM.arm_joints
    )
    path = tmp_path / "cal.json"
    ActuatorCalibration(entries).save(path)
    return str(path)


def _settle(backend: ServoEffectsBackend, target: np.ndarray, steps: int = 90) -> np.ndarray:
    """Ramp the command to `target` over a third of the steps, then hold."""
    start = backend.command[:N_ARM]
    for k in range(steps):
        goal = start + (target - start) * min(1.0, (k + 1) / (steps / 3))
        backend.apply_delta(goal - backend.command[:N_ARM])
        state = backend.step()
    return state.qpos[:N_ARM]


def test_mujoco_arm_lands_two_bands_apart_from_opposite_sides(tmp_path, rng):
    """With the measured stiffness and a band, from-above and from-below arrivals differ by about two bands."""
    band = np.deg2rad(1.5)
    sim = SimConfig(calibration_path=_calibration_file(tmp_path, band))
    params = StictionParams.from_calibration(ActuatorCalibration.load(sim.calibration_path), SIM.arm_joints)
    backend = ServoEffectsBackend(MujocoSo101Backend(sim), params, PLAY)
    home = backend.reset(rng).q_cmd[:N_ARM]
    lift = 1
    offset = np.zeros(N_ARM)
    offset[lift] = np.deg2rad(8.0)
    _settle(backend, home + offset)
    from_plus = _settle(backend, home)[lift]
    _settle(backend, home - offset)
    from_minus = _settle(backend, home)[lift]
    assert from_plus - from_minus == pytest.approx(2 * band, rel=0.25)
    backend.close()


def test_factory_wraps_with_servo_effects_inside_the_perturbation(tmp_path):
    """Servo effects act on what reaches the servo, so they sit inside the delay/rounding wrapper."""
    options = EnvOptions(
        calibration_path=_calibration_file(tmp_path, 0.02),
        perturbation=PerturbationConfig(delay_steps=1),
        servo=ServoEffectsConfig(stiction=True, relative_clip_rad=0.17),
    )
    env = make_reach_env(PRESETS["residual_ik"], randomize=False, options=options)
    assert isinstance(env.backend, PerturbedBackend)
    assert isinstance(env.backend.inner, ServoEffectsBackend)
    obs, _ = env.reset(seed=0)
    env.step(np.zeros(N_ARM, dtype=np.float32))
    assert np.all(np.isfinite(obs))
    env.close()


def test_stiction_without_calibration_is_refused():
    """The band comes from the calibration file, so asking for stiction without one is an error."""
    with pytest.raises(ValueError, match="calibration"):
        make_reach_env(PRESETS["residual_ik"], randomize=False, options=EnvOptions(servo=PLAY))


def test_env_options_with_servo_effects_round_trip_and_old_configs_load():
    """Saved run configs keep the servo settings; configs saved before they existed get the defaults."""
    options = EnvOptions(servo=ServoEffectsConfig(stiction=True, stiction_scale_range=(0.5, 1.5), relative_clip_rad=0.17))
    preset = ExperimentPreset(env=options)
    assert ExperimentPreset.from_dict(json.loads(json.dumps(preset.to_dict()))) == preset
    old = preset.to_dict()
    del old["env"]["servo"]
    assert ExperimentPreset.from_dict(old).env.servo == ServoEffectsConfig()


def test_command_window_tightens_the_limits_like_the_real_backend(rng):
    """With a window, commands clip at the tighter of the inner limits and the window."""
    window = tuple((-0.1, 0.05) for _ in range(N_ARM))
    backend = ServoEffectsBackend(IdealArm(), _params(), PLAY, command_window=window)
    backend.reset(rng)
    limits = backend.joint_limits()
    np.testing.assert_allclose(limits.lower, -0.1)
    np.testing.assert_allclose(limits.upper, 0.05)
    _move(backend, 0.5)
    np.testing.assert_allclose(backend.command[:N_ARM], 0.05)

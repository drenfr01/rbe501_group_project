"""Joint-torque analysis of SO-101 pick-and-place (RBE501 dynamics), in Lynch & Park notation.

Rigid-body dynamics of the arm joints θ ∈ R^5 (pan, lift, elbow, wrist flex, wrist roll), with every term written as
the torque the motors must supply:

    τ_act = M(θ)θ̈ + c(θ, θ̇) + g(θ) + τ_passive + τ_contact,      τ_contact = -J^T F (cube and other contacts)

A held cube of mass m whose center sits at the pinch point adds, when the arm is still, the payload torque

    τ_payload = J_v(θ)^T (-m g_s) = m b(θ),      b(θ) = J_v(θ)^T (-g_s)   (N·m per kg)

with J_v the linear (space) Jacobian of the pinch point and g_s = (0, 0, -9.81). The STS3215 servos are position
servos, τ = K_p (θ_cmd - θ) - K_d θ̇, with no gravity compensation, so at rest the arm settles short of its command by

    Δθ = θ_cmd - θ = K_p^-1 (g(θ) + m b(θ))        and the pinch point drops by  Δz ≈ J_v,z Δθ = b·Δθ / |g_s|

A joint whose |g_i + m b_i| exceeds its limit L cannot hold the cube at all; the largest mass a posture holds is
m_max = min_i (L - sign(b_i) g_i) / |b_i|.

Contents: `torque_terms` (split of MuJoCo's equation of motion), `ArmStatics` (g, b, sag, capacity at any posture),
`capacity_map`, `record_episode` -> `Trace` (+ `TraceSummary`), and `FeedforwardPickPlace` (scripted controller that
pre-tilts its command by K_p^-1 τ_hold, i.e. model-based gravity compensation with the cube mass known).
"""

from dataclasses import dataclass
from typing import Any, Callable, Self

import gymnasium as gym
import mujoco
import numpy as np

from rbe_501_so101.config import GripperConfig, SimConfig
from rbe_501_so101.control.scripted_pick_lift import Phase, PhaseTarget
from rbe_501_so101.envs.grasp_model import FIXED_PAD_GEOM, MOVING_PAD_GEOM
from rbe_501_so101.pick_place.env import PlacePhase, ScriptedPickPlace, ScriptedPickPlaceConfig

PAN, LIFT, ELBOW, WRIST_FLEX, WRIST_ROLL = range(5)
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

STALL_NM = SimConfig().stall_torque_nm
"""Torque cap used in the sim (STS3215 12 V stall: 30 kg·cm = 2.94 N·m)."""
RATED_NM = 0.98
"""STS3215 12 V rated (continuous) torque: 10 kg·cm."""
SATURATED = 0.99
"""|τ_act| above this fraction of the cap counts as saturated."""
HOLD_STEPS = 20
"""Hold window: the last steps of CARRY, when the arm waits above the goal with the cube."""
PLACE_STEPS = 10
"""Place window: the last steps of LOWER, the cube still held just above the table at the goal."""
PAD_MU = GripperConfig().pad_friction[0]
"""Sliding friction coefficient of the finger pads."""
HOLDING_PHASES = (Phase.LIFT, PlacePhase.CARRY, PlacePhase.LOWER)
"""Phases in which the scripted controller carries the cube."""


# ================================================================ equation of motion


@dataclass(frozen=True)
class TorqueTerms:
    """Arm-joint torques (N·m) of τ_act = Mθ̈ + c + g + τ_passive + τ_contact; arrays of shape (5,) or (T, 5)."""

    actuator: np.ndarray
    inertial: np.ndarray
    coriolis: np.ndarray
    gravity: np.ndarray
    passive: np.ndarray
    contact: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        """Sum of the load terms minus the actuator torque (zero up to round-off)."""
        return self.inertial + self.coriolis + self.gravity + self.passive + self.contact - self.actuator

    @classmethod
    def stack(cls, terms: list[Self]) -> Self:
        """Stack per-step terms into (T, 5) arrays."""
        return cls(**{k: np.stack([getattr(t, k) for t in terms]) for k in cls.__dataclass_fields__})


def gravity_torque(model: mujoco.MjModel, scratch: mujoco.MjData, qpos: np.ndarray) -> np.ndarray:
    """g over all DOFs at `qpos`: RNE with θ̇ = 0, θ̈ = 0 on a scratch copy."""
    scratch.qpos[:] = qpos
    scratch.qvel[:] = 0.0
    mujoco.mj_fwdPosition(model, scratch)
    g = np.zeros(model.nv)
    mujoco.mj_rne(model, scratch, 0, g)
    return g


def torque_terms(model: mujoco.MjModel, data: mujoco.MjData, scratch: mujoco.MjData, dof: np.ndarray) -> TorqueTerms:
    """Split MuJoCo's M q̈ + bias = τ_act + τ_passive_frc + τ_constraint at the current state, for the DOFs `dof`."""
    inertial = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, inertial, data.qacc)
    g = gravity_torque(model, scratch, data.qpos)
    return TorqueTerms(
        actuator=data.qfrc_actuator[dof].copy(),
        inertial=inertial[dof],
        coriolis=(data.qfrc_bias - g)[dof],
        gravity=g[dof],
        passive=-data.qfrc_passive[dof],
        contact=-data.qfrc_constraint[dof],
    )


# ================================================================ statics: g(θ), b(θ), sag, capacity


class ArmStatics:
    """Static joint loads of the env's arm at any posture: gravity g(θ) and payload torque per kg b(θ)."""

    def __init__(self, env: gym.Env) -> None:
        backend = env.unwrapped.backend
        cfg = env.unwrapped.config
        self.model, self._ids = backend.model, backend.ids
        self._scratch = mujoco.MjData(self.model)
        self._scratch.qpos[:] = backend.data.qpos
        self.dof = self._ids.arm_dof_adr
        self.kp = self.model.actuator_gainprm[self._ids.actuator_ids[: self._ids.n_arm], 0].copy()
        self._p_b = np.asarray(cfg.gripper.pinch_point_b(cfg.object.side_m))
        self._hand = self._ids.payload_body

    def loads(self, q: np.ndarray, point: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(g(θ), b(θ)) on the arm joints for robot joints q (arm + gripper); b for a mass at `point` in {s}
        (default: the nominal pinch point, where a centered cube's center sits)."""
        g, jac = self.loads_and_jacobian(q, point)
        return g, jac.T @ -self.model.opt.gravity

    def loads_and_jacobian(self, q: np.ndarray, point: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(g(θ), J_v(θ)): arm gravity torque and the 3x5 linear Jacobian of `point` (default pinch point)."""
        qpos = self._scratch.qpos.copy()
        qpos[self._ids.qpos_adr] = q
        g = gravity_torque(self.model, self._scratch, qpos)
        return g[self.dof], self._point_jacobian(self.pinch_point() if point is None else point)

    def pinch_point(self) -> np.ndarray:
        """Pinch point p_s = p_tcp + R_sb p_b at the scratch posture."""
        site = self._ids.tcp_site
        return self._scratch.site_xpos[site] + self._scratch.site_xmat[site].reshape(3, 3) @ self._p_b

    def sag(self, q: np.ndarray, mass_kg: float) -> np.ndarray:
        """Static servo error Δθ = K_p^-1 (g + m b) the arm settles at."""
        g, b = self.loads(q)
        return (g + mass_kg * b) / self.kp

    def capacity(self, q: np.ndarray, limit_nm: float) -> tuple[float, int]:
        """Largest held mass with every |g_i + m b_i| <= limit, and the joint that limits it."""
        g, b = self.loads(q)
        with np.errstate(divide="ignore"):
            per_joint = np.where(np.abs(b) > 1e-9, (limit_nm - np.sign(b) * g) / np.abs(b), np.inf)
        joint = int(np.argmin(per_joint))
        return max(0.0, float(per_joint[joint])), joint

    def _point_jacobian(self, point: np.ndarray) -> np.ndarray:
        """J_v (3x5) of a point in {s} rigidly attached to the hand, at the scratch posture."""
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self._scratch, jacp, None, point, self._hand)
        return jacp[:, self.dof]


@dataclass(frozen=True)
class CapacityMap:
    """Largest static payload (kg) over a grid of reach r (columns) and pinch height h (rows), top-down grasp."""

    radii_m: np.ndarray
    heights_m: np.ndarray
    limit_nm: float
    m_max_kg: np.ndarray
    joint: np.ndarray


def capacity_map(env: gym.Env, radii: tuple[float, ...], heights: tuple[float, ...], limit_nm: float) -> CapacityMap:
    """m_max(r, h) in front of the robot (azimuth 0), heights above the table-resting cube center."""
    statics, solver = ArmStatics(env), ScriptedPickPlace.for_env(env)
    backend = env.unwrapped.backend
    q = backend.grasp_state().robot.qpos.copy()
    m_max, joints = np.zeros((len(heights), len(radii))), np.zeros((len(heights), len(radii)), dtype=int)
    for i, h in enumerate(heights):
        for j, r in enumerate(radii):
            goal = np.array([*(backend.pan_axis_xy + (r, 0.0)), backend.rest_z + h])
            q[:-1] = solver._ik(goal, q, top_down=True)
            m_max[i, j], joints[i, j] = statics.capacity(q, limit_nm)
    return CapacityMap(np.asarray(radii), np.asarray(heights), limit_nm, m_max, joints)


# ================================================================ episode recording


@dataclass(frozen=True)
class TraceSummary:
    """Per-joint (5,) and scalar statistics of one episode.

    Whole episode: peak and RMS |τ_act|, fraction of steps at the stall cap, mechanical work Σ|τ θ̇|Δt (J), and the
    copper-loss proxy Σ τ² Δt (heat in the servo grows with current², and current ∝ torque).
    Hold window (end of CARRY): servo error θ_cmd - θ measured and predicted K_p^-1(g + m b); the correction the
    controller added on top of the IK target, θ_cmd - θ*; contact torque vs. the payload model m b; how far the cube
    hangs below the carry height (drop = arm sag of the pinch point + slip of the cube in the grip) and the J_v Δθ
    prediction of the arm part; pad squeeze ΣN and the friction margin μ ΣN / (m g) (> 1: friction can hold the cube).
    Place window (end of LOWER): the same correction and sag, how far sag alone moves the pinch point sideways,
    ||J_v,xy Δθ||, and the measured horizontal cube-goal offset just before release.
    Residual: mean |executed - scripted| joint step (rad) the RL policy added, per phase.
    """

    peak_nm: np.ndarray
    rms_nm: np.ndarray
    saturated_frac: np.ndarray
    work_j: np.ndarray
    heat_n2m2s: np.ndarray
    sag_measured: np.ndarray
    sag_predicted: np.ndarray
    correction: np.ndarray
    contact_nm: np.ndarray
    payload_model_nm: np.ndarray
    drop_measured_m: float
    drop_arm_m: float
    slip_m: float
    drop_predicted_m: float
    grip_force_n: float
    grip_margin: float
    place_correction: np.ndarray
    place_sag_measured: np.ndarray
    place_sag_predicted: np.ndarray
    place_shift_predicted_m: float
    place_offset_m: float
    residual_by_phase: dict[str, np.ndarray]
    success: bool
    error_m: float


@dataclass(frozen=True)
class Trace:
    """Everything recorded at each control step of one episode; joint arrays are (T, 5)."""

    label: str
    mass_kg: float
    dt_s: float
    phase: np.ndarray
    grasped: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    q_cmd: np.ndarray
    q_target: np.ndarray
    terms: TorqueTerms
    payload_model: np.ndarray
    jac: np.ndarray
    sag_predicted: np.ndarray
    cube_p: np.ndarray
    pinch_p: np.ndarray
    p_goal: np.ndarray
    carry_z: float
    grip_force_n: np.ndarray
    residual: np.ndarray
    success: bool
    error_m: float

    @property
    def b(self) -> np.ndarray:
        """Payload torque per kg b = J_v^T (-g_s), (T, 5)."""
        return G_S * self.jac[:, 2, :]

    @property
    def place(self) -> np.ndarray:
        """Mask of the last PLACE_STEPS of LOWER in which the cube is grasped."""
        return self._last(PlacePhase.LOWER, PLACE_STEPS)

    def _last(self, phase: str, n: int) -> np.ndarray:
        """Mask of the last n steps of a phase in which the cube is grasped."""
        steps = np.flatnonzero(self.phase == phase)
        mask = np.zeros(len(self.phase), dtype=bool)
        mask[steps[-n:]] = True
        return mask & self.grasped

    @property
    def t_s(self) -> np.ndarray:
        """Time of each step."""
        return self.dt_s * np.arange(1, len(self.phase) + 1)

    @property
    def hold(self) -> np.ndarray:
        """Mask of the last HOLD_STEPS of CARRY in which the cube is grasped."""
        return self._last(PlacePhase.CARRY, HOLD_STEPS)

    def summary(self) -> TraceSummary:
        """Load over the episode, sag / drop / grip in the hold window, sideways shift in the place window."""
        tau, hold = self.terms.actuator, self.hold
        sag = (self.q_cmd - self.q)[hold].mean(axis=0)
        sag_model = self.sag_predicted[hold].mean(axis=0)
        return TraceSummary(**self._hold_and_load(tau, hold, sag, sag_model), **self._place_stats(),
                            residual_by_phase=self._residual_by_phase(), success=self.success, error_m=self.error_m)

    def _place_stats(self) -> dict[str, Any]:
        """Correction, sag, and the sideways shift sag causes, just before release."""
        place = self.place
        shift = np.einsum("tij,tj->ti", self.jac[:, :2, :], self.sag_predicted)
        return dict(
            place_correction=_window_mean(self.q_cmd - self.q_target, place),
            place_sag_measured=_window_mean(self.q_cmd - self.q, place),
            place_sag_predicted=_window_mean(self.sag_predicted, place),
            place_shift_predicted_m=_nanmean(np.linalg.norm(shift, axis=1)[place]),
            place_offset_m=_nanmean(np.linalg.norm(self.cube_p[:, :2] - self.p_goal[:2], axis=1)[place]),
        )

    def _residual_by_phase(self) -> dict[str, np.ndarray]:
        """Mean |residual| joint step (rad) per phase, in phase order."""
        order = dict.fromkeys(self.phase)
        return {p: np.abs(self.residual[self.phase == p]).mean(axis=0) for p in order}

    def _hold_and_load(self, tau: np.ndarray, hold: np.ndarray, sag: np.ndarray, sag_model: np.ndarray) -> dict:
        """Whole-episode load statistics and the hold-window sag, drop, and grip."""
        cube_z, pinch_z = self.cube_p[:, 2], self.pinch_p[:, 2]
        return dict(
            peak_nm=np.abs(tau).max(axis=0),
            rms_nm=np.sqrt((tau**2).mean(axis=0)),
            saturated_frac=(np.abs(tau) >= SATURATED * STALL_NM).mean(axis=0),
            work_j=np.abs(tau * self.qd).sum(axis=0) * self.dt_s,
            heat_n2m2s=(tau**2).sum(axis=0) * self.dt_s,
            sag_measured=sag,
            sag_predicted=sag_model,
            correction=(self.q_cmd - self.q_target)[hold].mean(axis=0),
            contact_nm=self.terms.contact[hold].mean(axis=0),
            payload_model_nm=self.payload_model[hold].mean(axis=0),
            drop_measured_m=_nanmean(self.carry_z - cube_z[hold]),
            drop_arm_m=_nanmean(self.carry_z - pinch_z[hold]),
            slip_m=_nanmean(pinch_z[hold] - cube_z[hold]),
            drop_predicted_m=_nanmean((self.b * self.sag_predicted).sum(axis=1)[hold]) / G_S,
            grip_force_n=_nanmean(self.grip_force_n[hold]),
            grip_margin=PAD_MU * _nanmean(self.grip_force_n[hold]) / (self.mass_kg * G_S),
        )


G_S = 9.81
"""|g_s| (m/s^2), MuJoCo's default gravity."""


def _window_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-joint mean over a window, NaN if the window is empty."""
    return x[mask].mean(axis=0) if mask.any() else np.full(x.shape[1], np.nan)


def _nanmean(x: np.ndarray) -> float:
    """Mean, or NaN for an empty window (cube never held during the hold)."""
    return float(np.mean(x)) if len(x) else float("nan")


@dataclass(frozen=True)
class _Step:
    """One control step's raw readings (internal to `record_episode`)."""

    phase: str
    grasped: bool
    q: np.ndarray
    qd: np.ndarray
    q_cmd: np.ndarray
    q_target: np.ndarray
    terms: TorqueTerms
    g: np.ndarray
    jac: np.ndarray
    cube_p: np.ndarray
    pinch_p: np.ndarray
    grip_force_n: float
    residual: np.ndarray


class _Recorder:
    """Reads torques, sag model, and grip force from the env after each step."""

    def __init__(self, env: gym.Env) -> None:
        self._env, self._backend = env.unwrapped, env.unwrapped.backend
        self.statics = ArmStatics(env)
        self._scratch = mujoco.MjData(self._backend.model)
        model = self._backend.model
        self._pads = {model.geom(FIXED_PAD_GEOM).id, model.geom(MOVING_PAD_GEOM).id}

    def read(self, controller: ScriptedPickPlace, residual: np.ndarray) -> _Step:
        """Snapshot after a step; `controller` gives the current phase and IK target θ*."""
        s = self._backend.grasp_state()
        n = self._backend.ids.n_arm
        g, jac = self.statics.loads_and_jacobian(s.robot.qpos, s.obj.p_s if s.grasped else None)
        terms = torque_terms(self._backend.model, self._backend.data, self._scratch, self.statics.dof)
        return _Step(str(controller.phase), s.grasped, s.robot.qpos[:n], s.robot.qvel[:n], s.robot.q_cmd[:n],
                     controller._target.q_arm.copy(), terms, g, jac, s.obj.p_s.copy(), s.p_pinch.copy(),
                     self._grip_force(), residual)

    def _grip_force(self) -> float:
        """Sum of the normal contact forces between the finger pads and the cube (N)."""
        model, data, force = self._backend.model, self._backend.data, np.zeros(6)
        total = 0.0
        for i, c in enumerate(data.contact[: data.ncon]):
            pair = {int(c.geom1), int(c.geom2)}
            if self._env._cube_geom in pair and pair & self._pads:
                mujoco.mj_contactForce(model, data, i, force)
                total += force[0]
        return total


def record_episode(env: gym.Env, policy: Callable[[np.ndarray], np.ndarray], controller: ScriptedPickPlace,
                   seed: int, label: str, options: dict[str, Any] | None = None) -> Trace:
    """Roll out one episode and record torques, sag, and grip force at every control step.

    `controller` is the scripted controller whose phase and IK target θ* are logged; it is `policy` itself for a
    scripted run, or the one inside the residual wrapper for a trained policy.
    """
    recorder = _Recorder(env)
    obs, _ = env.reset(seed=seed, options=options)
    getattr(policy, "reset", lambda: None)()
    steps, truncated, info = [], False, {}
    dq_max = env.unwrapped.config.task.delta_q_max_rad
    n = env.unwrapped.backend.ids.n_arm
    while not truncated:
        base = env.base_action().copy() if hasattr(env, "base_action") else None
        obs, _, _, truncated, info = env.step(policy(obs))
        residual = np.zeros(n) if base is None else (env.last_executed_action - base)[:n] * dq_max
        steps.append(recorder.read(controller, residual))
    return _build_trace(env, steps, recorder.statics, label, info)


def _build_trace(env: gym.Env, steps: list[_Step], statics: ArmStatics, label: str, info: dict) -> Trace:
    """Stack the per-step readings and add the payload and sag models."""
    u = env.unwrapped
    mass = u.backend.object_mass_kg
    grasped = np.array([s.grasped for s in steps])
    jac, g = np.stack([s.jac for s in steps]), np.stack([s.g for s in steps])
    payload = mass * G_S * jac[:, 2, :] * grasped[:, None]
    carry_z = float(u.p_goal[2] + ScriptedPickPlaceConfig().carry_height_m)
    return Trace(label, mass, u.control_dt, np.array([s.phase for s in steps]), grasped,
                 *(np.stack([getattr(s, k) for s in steps]) for k in ("q", "qd", "q_cmd", "q_target")),
                 TorqueTerms.stack([s.terms for s in steps]), payload, jac, (g + payload) / statics.kp,
                 *(np.stack([getattr(s, k) for s in steps]) for k in ("cube_p", "pinch_p")), u.p_goal.copy(), carry_z,
                 np.array([s.grip_force_n for s in steps]), np.stack([s.residual for s in steps]),
                 bool(info["is_success"]), float(info["goal_distance_m"]))


# ================================================================ model-based baseline


class FeedforwardPickPlace(ScriptedPickPlace):
    """Scripted pick-and-place with model-based gravity compensation through the position servos.

    Every joint target θ* is replaced by θ* + K_p^-1 (g(θ*) + m b(θ*)), with m the true cube mass (an oracle) while
    the cube is carried, so the servo's steady error lands the arm on θ*. Saturated joints cannot be compensated.
    """

    statics: ArmStatics
    payload_only: bool = False

    @classmethod
    def for_env(cls, env: gym.Env, cfg: ScriptedPickPlaceConfig = ScriptedPickPlaceConfig(),
                payload_only: bool = False) -> Self:
        """Controller for `env`; `payload_only` leaves the arm's own-weight sag (the script was tuned with it)."""
        controller = super().for_env(env, cfg)
        controller.statics, controller.payload_only = ArmStatics(env), payload_only
        controller._backend = env.unwrapped.backend
        return controller

    def _track(self, obs: np.ndarray, target: PhaseTarget) -> np.ndarray:
        """Track the gravity-compensated target instead of θ*."""
        q = np.append(target.q_arm, target.gripper_rad)
        mass = self._backend.object_mass_kg if self._phase in HOLDING_PHASES else 0.0
        g, b = self.statics.loads(q)
        offset = (mass * b if self.payload_only else g + mass * b) / self.statics.kp
        shifted = PhaseTarget(target.q_arm + offset, target.gripper_rad, target.arm_rate_rad, target.gripper_rate_rad)
        return super()._track(obs, shifted)

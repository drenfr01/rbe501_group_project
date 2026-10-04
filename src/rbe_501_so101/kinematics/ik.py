"""Position-only damped least-squares inverse kinematics for the SO-101 gripper site."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

import mujoco
import numpy as np

from rbe_501_so101.config import IKConfig, SimConfig

CARTESIAN_DIM = 3
PAN_INDEX = 0
POSTURE_SLICE = slice(1, 4)
PAN_PROBE_RAD = 0.5


@dataclass(frozen=True)
class IKResult:
    """Outcome of an IK solve: full joint vector (arm + gripper), error, iterations."""

    q: np.ndarray
    error_m: float
    iterations: int

    def converged(self, tolerance_m: float) -> bool:
        """Return True if the residual position error is within tolerance."""
        return self.error_m < tolerance_m


class PositionIKSolver:
    """Solves world XYZ targets for the arm joints on a scratch kinematic state.

    `solve` first iterates from the caller's start pose, then restarts from
    seed postures with the pan joint aimed at the target (front or back), and
    returns the best result. Orientation is unconstrained, the gripper angle
    is left unchanged, and physics is never advanced.
    """

    def __init__(self, model: mujoco.MjModel, sim_cfg: SimConfig, cfg: IKConfig) -> None:
        self._model = model
        self._cfg = cfg
        self._data = mujoco.MjData(model)
        joint_ids = [model.joint(n).id for n in sim_cfg.joints]
        n_arm = len(sim_cfg.arm_joints)
        self._qpos_adr = model.jnt_qposadr[joint_ids]
        self._arm_dof_adr = model.jnt_dofadr[joint_ids[:n_arm]]
        self._arm_range = model.jnt_range[joint_ids[:n_arm]]
        self._site_id = model.site(sim_cfg.tcp_site).id
        self._jac = np.zeros((CARTESIAN_DIM, model.nv))
        self._axis_xy = self._pan_axis_xy(sim_cfg.radial_axis_joint)
        self._pan_sign = self._measure_pan_sign(np.deg2rad(sim_cfg.initial_deg))

    @classmethod
    def from_config(cls, sim_cfg: SimConfig, cfg: IKConfig) -> Self:
        """Build a solver with its own copy of the configured model."""
        return cls(mujoco.MjModel.from_xml_path(str(sim_cfg.model_path)), sim_cfg, cfg)

    def forward(self, q: np.ndarray) -> np.ndarray:
        """Return the gripper-site world position for a full joint vector."""
        self._data.qpos[self._qpos_adr] = q
        mujoco.mj_kinematics(self._model, self._data)
        return self._data.site_xpos[self._site_id].copy()

    def solve(self, target: np.ndarray, q_init: np.ndarray) -> IKResult:
        """Return the first converged (or else the best) solution over all starts."""
        best: IKResult | None = None
        for start in self._starts(target, np.asarray(q_init, dtype=float)):
            result = self.solve_from(target, start)
            if best is None or result.error_m < best.error_m:
                best = result
            if best.converged(self._cfg.tolerance_m):
                break
        return best

    def solve_from(self, target: np.ndarray, q_init: np.ndarray) -> IKResult:
        """Iterate damped least squares from a single start pose."""
        q = np.asarray(q_init, dtype=float).copy()
        error = target - self.forward(q)
        iteration = 0
        while np.linalg.norm(error) >= self._cfg.tolerance_m and iteration < self._cfg.max_iterations:
            q = self._update(q, error)
            error = target - self.forward(q)
            iteration += 1
        return IKResult(q=q, error_m=float(np.linalg.norm(error)), iterations=iteration)

    def _starts(self, target: np.ndarray, q_init: np.ndarray) -> Iterator[np.ndarray]:
        """The caller's pose, then every (seed posture, pan candidate) pair."""
        yield q_init
        for posture in self._cfg.seed_postures_deg:
            for pan in self._pan_candidates(target):
                start = q_init.copy()
                start[PAN_INDEX] = pan
                start[POSTURE_SLICE] = np.deg2rad(posture)
                yield start

    def _pan_candidates(self, target: np.ndarray) -> list[float]:
        """Pan angles facing the target directly or from behind, within limits."""
        offset = np.asarray(target)[:2] - self._axis_xy
        facing = self._pan_sign * float(np.arctan2(offset[1], offset[0]))
        low, high = self._arm_range[PAN_INDEX]
        return [p for p in (facing, facing + np.pi, facing - np.pi) if low <= p <= high]

    def _update(self, q: np.ndarray, error: np.ndarray) -> np.ndarray:
        """One clipped damped-least-squares step on the arm joints."""
        mujoco.mj_comPos(self._model, self._data)
        mujoco.mj_jacSite(self._model, self._data, self._jac, None, self._site_id)
        jac = self._jac[:, self._arm_dof_adr]
        damping = self._cfg.damping**2 * np.eye(CARTESIAN_DIM)
        dq = self._cfg.step_gain * jac.T @ np.linalg.solve(jac @ jac.T + damping, error)
        dq = np.clip(dq, -self._cfg.max_step_rad, self._cfg.max_step_rad)
        n_arm = len(self._arm_dof_adr)
        q[:n_arm] = np.clip(q[:n_arm] + dq, self._arm_range[:, 0], self._arm_range[:, 1])
        return q

    def _pan_axis_xy(self, joint_name: str) -> np.ndarray:
        """World XY of the pan axis at the model's reference pose."""
        mujoco.mj_kinematics(self._model, self._data)
        return self._data.xanchor[self._model.joint(joint_name).id][:2].copy()

    def _measure_pan_sign(self, home: np.ndarray) -> float:
        """+1 if positive pan rotates the TCP counter-clockwise about +z, else -1."""
        azimuths = []
        for pan in (0.0, PAN_PROBE_RAD):
            q = home.copy()
            q[PAN_INDEX] = pan
            offset = self.forward(q)[:2] - self._axis_xy
            azimuths.append(np.arctan2(offset[1], offset[0]))
        return float(np.sign(azimuths[1] - azimuths[0]))

"""Damped least-squares IK for the gripper pinch point."""

from dataclasses import dataclass

import mujoco
import numpy as np

from rbe_501_so101.config import PinchIKConfig
from rbe_501_so101.envs.model_indices import ModelIndices

CARTESIAN_DIM = 3
TWIST_DIM = 6
ANGULAR = slice(0, 3)
LINEAR = slice(3, 6)
Z_COL = 2
PAN_INDEX = 0
POSTURE_SLICE = slice(1, 4)
DOWN_S = np.array([0.0, 0.0, -1.0])


def skew(v: np.ndarray) -> np.ndarray:
    """[v]: the skew-symmetric matrix with [v] w = v x w."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@dataclass(frozen=True)
class PinchIKResult:
    """Joint vector (arm + gripper), pinch-point error (m), and approach-axis tilt (rad)."""

    q: np.ndarray
    error_m: float
    tilt_rad: float
    top_down: bool
    cfg: PinchIKConfig

    def converged(self) -> bool:
        """Position within tolerance and, for top-down solves, the axis within tilt tolerance."""
        upright = not self.top_down or self.tilt_rad < self.cfg.tilt_tolerance_rad
        return self.error_m < self.cfg.tolerance_m and upright


class PinchIKSolver:
    """Solve pinch-point position with optional top-down orientation."""

    def __init__(self, model: mujoco.MjModel, ids: ModelIndices, cfg: PinchIKConfig, p_b: np.ndarray) -> None:
        self._model = model
        self._ids = ids
        self._cfg = cfg
        self._p_b = np.asarray(p_b, dtype=float)
        self._data = mujoco.MjData(model)
        self._arm_range = model.jnt_range[ids.arm_joint_ids]
        self._jac_v = np.zeros((CARTESIAN_DIM, model.nv))
        self._jac_w = np.zeros((CARTESIAN_DIM, model.nv))
        self._pan_axis_xy = self._pan_axis()

    def tcp_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(p_sb, R_sb) of the gripper site for a full joint vector."""
        self._data.qpos[self._ids.qpos_adr] = q
        mujoco.mj_kinematics(self._model, self._data)
        site = self._ids.tcp_site
        return self._data.site_xpos[site].copy(), self._data.site_xmat[site].reshape(3, 3).copy()

    def forward(self, q: np.ndarray) -> np.ndarray:
        """Pinch point in {s}: p_s = p_sb + R_sb p_b."""
        p_sb, R_sb = self.tcp_pose(q)
        return p_sb + R_sb @ self._p_b

    def solve(self, goal: np.ndarray, q_init: np.ndarray, top_down: bool) -> PinchIKResult:
        """Solve from `q_init`; if that fails, retry with the pan aimed at the goal."""
        best = self.solve_from(goal, q_init, top_down)
        for start in self._restarts(goal, q_init):
            if best.converged():
                break
            result = self.solve_from(goal, start, top_down)
            best = result if result.error_m < best.error_m else best
        return best

    def solve_from(self, goal: np.ndarray, q_init: np.ndarray, top_down: bool) -> PinchIKResult:
        """Iterate damped least squares from one start pose."""
        q = np.asarray(q_init, dtype=float).copy()
        for _ in range(self._cfg.max_iterations):
            twist = self._error_twist(q, goal)
            if self._result(q, twist, top_down).converged():
                break
            q = self._step(q, twist, top_down)
        return self._result(q, self._error_twist(q, goal), top_down)

    def _error_twist(self, q: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """V_e = [z_b x (-z_s); p_goal - p_pinch] at configuration q."""
        p_pinch = self.forward(q)
        z_b = self._data.site_xmat[self._ids.tcp_site].reshape(3, 3)[:, Z_COL]
        return np.concatenate([np.cross(z_b, DOWN_S), goal - p_pinch])

    def _jacobian(self) -> np.ndarray:
        """6 x n_arm pinch-point Jacobian [J_omega; J_v - [R_sb p_b] J_omega] at the last FK."""
        mujoco.mj_comPos(self._model, self._data)
        mujoco.mj_jacSite(self._model, self._data, self._jac_v, self._jac_w, self._ids.tcp_site)
        dofs = self._ids.arm_dof_adr
        J_w, J_v = self._jac_w[:, dofs], self._jac_v[:, dofs]
        r_s = self._data.site_xmat[self._ids.tcp_site].reshape(3, 3) @ self._p_b
        return np.vstack([J_w, J_v - skew(r_s) @ J_w])

    def _step(self, q: np.ndarray, twist: np.ndarray, top_down: bool) -> np.ndarray:
        """One clipped damped-least-squares update of the arm joints."""
        rows = slice(0, TWIST_DIM) if top_down else LINEAR
        J, V = self._jacobian()[rows], twist[rows]
        damping = self._cfg.damping**2 * np.eye(J.shape[0])
        dq = self._cfg.step_gain * J.T @ np.linalg.solve(J @ J.T + damping, V)
        dq = np.clip(dq, -self._cfg.max_step_rad, self._cfg.max_step_rad)
        n_arm = self._ids.n_arm
        q[:n_arm] = np.clip(q[:n_arm] + dq, self._arm_range[:, 0], self._arm_range[:, 1])
        return q

    def _result(self, q: np.ndarray, twist: np.ndarray, top_down: bool) -> PinchIKResult:
        """Package q with its position error and approach-axis tilt."""
        tilt = float(np.arcsin(min(1.0, np.linalg.norm(twist[ANGULAR]))))
        return PinchIKResult(q.copy(), float(np.linalg.norm(twist[LINEAR])), tilt, top_down, self._cfg)

    def _restarts(self, goal: np.ndarray, q_init: np.ndarray) -> list[np.ndarray]:
        """Seed posture with the pan pointed at the goal, trying both pan signs."""
        q_ref = np.asarray(q_init, dtype=float)
        offset = np.asarray(goal)[:2] - self._pan_axis_xy
        bearing = float(np.arctan2(offset[1], offset[0]))
        starts = []
        for pan in (bearing, -bearing):
            start = q_ref.copy()
            start[PAN_INDEX] = pan
            start[POSTURE_SLICE] = np.deg2rad(self._cfg.seed_posture_deg)
            starts.append(start)
        return starts

    def _pan_axis(self) -> np.ndarray:
        """World XY of the shoulder-pan axis at the reference pose."""
        mujoco.mj_kinematics(self._model, self._data)
        return self._data.xanchor[self._ids.arm_joint_ids[PAN_INDEX]][:2].copy()

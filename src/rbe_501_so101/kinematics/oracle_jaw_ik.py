"""Experimental position-first IK with a cube-face jaw-axis objective."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rbe_501_so101.kinematics.pinch_ik import PinchIKResult, PinchIKSolver, skew


@dataclass(frozen=True)
class JawIKConfig:
    """Fixed prototype weights and acceptance bounds; no arbitrary full-pose constraint."""

    axis_weight_m: float = .03
    approach_weight_m: float = .001
    damping: float = .005
    iterations: int = 200
    axis_tolerance_rad: float = np.deg2rad(5.)
    solve_axis_tolerance_rad: float = np.deg2rad(1.)
    maximum_tilt_rad: float = np.deg2rad(25.)


def nearest_face_axis(cube_yaw: float, bearing: float) -> np.ndarray:
    """Select the equivalent square-face normal nearest the nominal jaw direction."""
    yaw = bearing + (cube_yaw - bearing + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return np.array([np.cos(yaw), np.sin(yaw), 0.])


@dataclass(frozen=True)
class JawIKResult(PinchIKResult):
    """Report position, actual approach tilt and signed-axis alignment separately."""

    axis_error_rad: float
    jaw_cfg: JawIKConfig

    def converged(self) -> bool:
        """Require reachable position and jaw direction while limiting approach tilt."""
        return (self.error_m < self.cfg.tolerance_m
                and self.axis_error_rad < self.jaw_cfg.solve_axis_tolerance_rad
                and self.tilt_rad <= self.jaw_cfg.maximum_tilt_rad)


class JawIKSolver(PinchIKSolver):
    """Use three position and two jaw-axis constraints, with a weak approach preference."""

    def __init__(self, model, ids, cfg, p_b, cube_yaw: Callable[[], float], jaw_cfg=JawIKConfig()) -> None:
        """Keep an independent MuJoCo scratch state and an explicit oracle callback."""
        super().__init__(model, ids, cfg, p_b)
        self.cube_yaw, self.jaw_cfg = cube_yaw, jaw_cfg
        self.desired_axis: np.ndarray | None = None
        self.last_result: JawIKResult | None = None

    def reset_axis(self) -> None:
        """Read the initial cube orientation once for the next acquisition."""
        self.desired_axis = None
        self.last_result = None

    def axis_jacobian(self, axis: np.ndarray) -> np.ndarray:
        """d(axis)/dq = -[axis] J_omega using MuJoCo's site angular Jacobian."""
        return -skew(axis) @ self._jacobian()[:3]

    def _oriented_result(self, q: np.ndarray, goal: np.ndarray) -> JawIKResult:
        """Measure true angles with acos, including inverted approach configurations."""
        position = self.forward(q)
        rotation = self._data.site_xmat[self._ids.tcp_site].reshape(3, 3)
        axis_error = float(np.arccos(np.clip(rotation[:, 0] @ self.desired_axis, -1, 1)))
        tilt = float(np.arccos(np.clip(-rotation[2, 2], -1, 1)))
        return JawIKResult(q.copy(), float(np.linalg.norm(goal - position)), tilt,
                           True, self._cfg, axis_error, self.jaw_cfg)

    def _oriented_step(self, q: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Weighted DLS keeps Cartesian position more important than orientation."""
        position = self.forward(q)
        rotation = self._data.site_xmat[self._ids.tcp_site].reshape(3, 3)
        jac = self._jacobian()
        axis, approach = rotation[:, 0], rotation[:, 2]
        cfg = self.jaw_cfg
        error = np.concatenate([goal - position, cfg.axis_weight_m * (self.desired_axis - axis),
                                cfg.approach_weight_m * (np.array([0., 0., -1.]) - approach)])
        matrix = np.vstack([jac[3:], -cfg.axis_weight_m * skew(axis) @ jac[:3],
                            -cfg.approach_weight_m * skew(approach) @ jac[:3]])
        delta = matrix.T @ np.linalg.solve(matrix @ matrix.T + cfg.damping**2 * np.eye(9), error)
        delta = np.clip(self._cfg.step_gain * delta, -self._cfg.max_step_rad, self._cfg.max_step_rad)
        q[:self._ids.n_arm] = np.clip(q[:self._ids.n_arm] + delta, self._arm_range[:, 0], self._arm_range[:, 1])
        return q

    def _oriented_from(self, goal: np.ndarray, start: np.ndarray) -> JawIKResult:
        """Try one initial posture without changing the commanded gripper angle."""
        q = start.copy()
        for _ in range(self.jaw_cfg.iterations):
            result = self._oriented_result(q, goal)
            if result.converged():
                return result
            q = self._oriented_step(q, goal)
        return self._oriented_result(q, goal)

    def solve(self, goal: np.ndarray, q_init: np.ndarray, top_down: bool) -> PinchIKResult:
        """Preserve free-position raising/lifting; apply the oracle objective to acquisition."""
        if not top_down:
            return super().solve(goal, q_init, False)
        nominal = super().solve(goal, q_init, True)
        if self.desired_axis is None:
            _, rotation = self.tcp_pose(nominal.q)
            bearing = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            self.desired_axis = nearest_face_axis(self.cube_yaw(), bearing)
        candidates = [nominal.q, q_init.copy(), *self._restarts(goal, q_init)]
        best = self._oriented_from(goal, candidates[0])
        for start in candidates[1:]:
            if best.converged():
                break
            result = self._oriented_from(goal, start)
            score = lambda r: r.error_m / self._cfg.tolerance_m + r.axis_error_rad / self.jaw_cfg.axis_tolerance_rad
            best = result if score(result) < score(best) else best
        self.last_result = best
        return best

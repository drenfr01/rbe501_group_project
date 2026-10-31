"""Pre-flight table check of a joint-space path, run before any hardware script enables torque."""

import math

import numpy as np

from rbe_501_so101.config import PathCheckConfig
from rbe_501_so101.hardware.safety import SafetyAbort, SafetyMonitor

MM_PER_M = 1000.0


def segment_points(start: np.ndarray, end: np.ndarray, step_rad: float) -> np.ndarray:
    """Joint-space line from `start` to `end`, no joint moving more than `step_rad` per point."""
    n = max(1, math.ceil(float(np.abs(end - start).max()) / step_rad))
    return start + np.outer(np.linspace(0.0, 1.0, n + 1), end - start)


def plan_path(start: np.ndarray, poses: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    """Waypoints: the start, every pose in order, then the start again."""
    return [start, *poses, start]


def preflight(safety: SafetyMonitor, path: list[np.ndarray], cfg: PathCheckConfig) -> float:
    """Smallest table margin (m) along the path, with and without sag; raise if it goes too low.

    A start already inside the clearance (an arm resting low) is allowed, but
    nothing on the path may go lower than the start does.
    """
    sag = np.deg2rad(np.append(cfg.sag_allowance_deg, 0.0))
    floor = min(0.0, safety.clearance_margin(path[0]), safety.clearance_margin(path[0] + sag))
    worst = math.inf
    for start, end in zip(path[:-1], path[1:], strict=True):
        for q in segment_points(start, end, math.radians(cfg.step_deg)):
            worst = min(worst, safety.clearance_margin(q), safety.clearance_margin(q + sag))
    if worst < floor:
        raise SafetyAbort(f"path comes {MM_PER_M * -worst:.0f} mm inside the table clearance")
    return worst

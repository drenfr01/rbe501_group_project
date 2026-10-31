"""Plan step 10: does the calibrated sim reproduce the real arm's errors, per joint and arrival direction?"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rbe_501_so101.hardware.direction_errors import DirectionalError, VisitError

WITH, AGAINST, UNLOADED = "with gravity", "against gravity", "no gravity load"


@dataclass(frozen=True)
class GateRow:
    """Real vs sim error (sim degrees, along gravity) for one source, joint, and arrival direction."""

    source: str
    joint: str
    direction: str
    n: int
    real_mean_deg: float
    sim_mean_deg: float
    mean_abs_diff_deg: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    """All rows and the verdict."""

    rows: tuple[GateRow, ...]
    tolerance_deg: float

    @property
    def passed(self) -> bool:
        """True if every row is within the tolerance."""
        return all(r.passed for r in self.rows)


def directional_rows(
    source: str, real: Sequence[DirectionalError], sim: Sequence[DirectionalError], tolerance_deg: float
) -> list[GateRow]:
    """With- and against-gravity rows per joint, pairing real and sim by pose and joint."""
    sim_by = {(e.pose_id, e.joint): e for e in sim}
    pairs = [(r, sim_by[(r.pose_id, r.joint)]) for r in real if (r.pose_id, r.joint) in sim_by]
    rows = []
    for joint in dict.fromkeys(r.joint for r, _ in pairs):
        mine = [(r, s) for r, s in pairs if r.joint == joint]
        rows.append(_row(source, joint, WITH, [(r.with_gravity_deg, s.with_gravity_deg) for r, s in mine], tolerance_deg))
        rows.append(
            _row(source, joint, AGAINST, [(r.against_gravity_deg, s.against_gravity_deg) for r, s in mine], tolerance_deg)
        )
    return rows


def visit_rows(
    source: str, real: Sequence[VisitError], sim: Sequence[VisitError], tolerance_deg: float
) -> list[GateRow]:
    """Pose-check rows per joint and arrival direction, pairing real and sim by visit and joint."""
    sim_by = {(v.visit, v.joint): v for v in sim}
    pairs = [(r, sim_by[(r.visit, r.joint)]) for r in real if (r.visit, r.joint) in sim_by]
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r, s in pairs:
        groups.setdefault((r.joint, _direction(r.arrived_with_gravity)), []).append((r.error_deg, s.error_deg))
    return [_row(source, joint, direction, values, tolerance_deg) for (joint, direction), values in groups.items()]


def _direction(arrived_with_gravity: bool | None) -> str:
    """Row label for an arrival."""
    if arrived_with_gravity is None:
        return UNLOADED
    return WITH if arrived_with_gravity else AGAINST


def _row(source: str, joint: str, direction: str, pairs: list[tuple[float, float]], tolerance_deg: float) -> GateRow:
    """Means and mean absolute difference of (real, sim) pairs."""
    real, sim = np.array(pairs).T
    diff = float(np.mean(np.abs(real - sim)))
    return GateRow(source, joint, direction, len(pairs), float(real.mean()), float(sim.mean()), diff, diff <= tolerance_deg)

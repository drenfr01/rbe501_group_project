"""Phase 1 physical-reference readings (level links, mechanical stops) and offset computation."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferenceReading:
    """The arm held at a physical reference: a note, the LeRobot reading, and the mapped sim angles (deg)."""

    note: str
    lerobot_deg: dict[str, float]
    sim_deg: tuple[float, ...]
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ReferenceRepository:
    """Append-only JSON list of reference readings."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, reading: ReferenceReading) -> None:
        """Add one reading and rewrite the file."""
        readings = [*self.load(), reading]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(r) for r in readings], indent=2))

    def load(self) -> list[ReferenceReading]:
        """All readings (empty if the file does not exist yet)."""
        if not self.path.exists():
            return []
        rows = json.loads(self.path.read_text())
        return [ReferenceReading(r["note"], r["lerobot_deg"], tuple(r["sim_deg"]), r["recorded_at"]) for r in rows]


def offset_from_reference(lerobot_deg: float, sim_reference_rad: float, sign: float) -> float:
    """Offset (rad) so that `sign * deg2rad(lerobot_deg) + offset` equals the reference's sim angle."""
    return float(sim_reference_rad - sign * np.deg2rad(lerobot_deg))

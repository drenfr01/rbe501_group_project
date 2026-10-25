"""Measured actuator stiffness and friction, applied on top of the placeholder MJCF."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import mujoco
import numpy as np

from rbe_501_so101.config import SimConfig

KP_GAIN_COL = 0
KP_BIAS_COL = 1
KV_BIAS_COL = 2


@dataclass(frozen=True)
class JointActuatorCalibration:
    """Measured values for one joint; `None` keeps the MJCF value."""

    joint: str
    kp_nm_per_rad: float | None = None
    frictionloss_nm: float | None = None


@dataclass(frozen=True)
class ActuatorCalibration:
    """Per-joint stiffness and Coulomb friction, plus a note on where they came from.

    Changing `kp` rescales the actuator's velocity gain by `sqrt(kp_new / kp_old)`,
    which keeps the MJCF's `dampratio` (fixed at compile time) unchanged.
    """

    joints: tuple[JointActuatorCalibration, ...]
    source: str = ""

    def apply(self, model: mujoco.MjModel, sim_cfg: SimConfig = SimConfig()) -> None:
        """Write the measured values into the model in place."""
        for entry in self.joints:
            if entry.kp_nm_per_rad is not None:
                _set_kp(model, model.actuator(entry.joint + sim_cfg.actuator_suffix).id, entry.kp_nm_per_rad)
            if entry.frictionloss_nm is not None:
                model.dof_frictionloss[model.joint(entry.joint).dofadr[0]] = entry.frictionloss_nm

    def kp_for(self, joints: tuple[str, ...], default: np.ndarray) -> np.ndarray:
        """Stiffness per joint: measured where available, else `default`."""
        measured = {e.joint: e.kp_nm_per_rad for e in self.joints if e.kp_nm_per_rad is not None}
        return np.array([measured.get(j, d) for j, d in zip(joints, default, strict=True)], dtype=float)

    def save(self, path: Path) -> None:
        """Write as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a calibration written by `save`."""
        data = json.loads(Path(path).read_text())
        return cls(joints=tuple(JointActuatorCalibration(**j) for j in data["joints"]), source=data.get("source", ""))


def _set_kp(model: mujoco.MjModel, actuator: int, kp: float) -> None:
    """Set a position actuator's stiffness, keeping its damping ratio."""
    kp_old = float(model.actuator_gainprm[actuator, KP_GAIN_COL])
    model.actuator_gainprm[actuator, KP_GAIN_COL] = kp
    model.actuator_biasprm[actuator, KP_BIAS_COL] = -kp
    model.actuator_biasprm[actuator, KV_BIAS_COL] *= np.sqrt(kp / kp_old)

"""Abort checks run on every hardware control step: table clearance, motor status, loop timing."""

from typing import Self

import mujoco
import numpy as np

from rbe_501_so101.config import SafetyConfig, SimConfig
from rbe_501_so101.envs.model_indices import ModelIndices, descendant_body_ids, table_clearance_z
from rbe_501_so101.hardware.types import MotorStatus


class SafetyAbort(RuntimeError):
    """Raised when a hardware run must stop; the message names the violated rule."""


class SafetyMonitor:
    """Stateful safety checks for one hardware session.

    Clearance uses the body origins below `clearance_root_body` plus the
    gripper site, against the table-top height treated as an infinite plane
    (conservative where the arm hangs past the table edge).
    """

    def __init__(self, model: mujoco.MjModel, ids: ModelIndices, sim_cfg: SimConfig, cfg: SafetyConfig) -> None:
        self._model = model
        self._ids = ids
        self._cfg = cfg
        self._data = mujoco.MjData(model)
        self._bodies = np.array(sorted(descendant_body_ids(model, cfg.clearance_root_body)))
        self._z_min = table_clearance_z(model, sim_cfg.table_top_geom, cfg.table_clearance_m)
        self._consecutive_overruns = 0

    @classmethod
    def from_model(cls, model: mujoco.MjModel, sim_cfg: SimConfig, cfg: SafetyConfig = SafetyConfig()) -> Self:
        """Monitor sharing an existing model (it keeps its own `MjData`)."""
        return cls(model, ModelIndices.from_model(model, sim_cfg), sim_cfg, cfg)

    @property
    def status_check_every_steps(self) -> int:
        """Control steps between motor status reads."""
        return self._cfg.status_check_every_steps

    def lowest_point_z(self, q: np.ndarray) -> float:
        """Lowest monitored world Z (m) for a full joint vector."""
        self._data.qpos[self._ids.qpos_adr] = q
        mujoco.mj_kinematics(self._model, self._data)
        body_z = self._data.xpos[self._bodies, 2]
        return float(min(body_z.min(), self._data.site_xpos[self._ids.tcp_site, 2]))

    def clearance_margin(self, q: np.ndarray) -> float:
        """Metres the lowest monitored point sits above the table clearance (negative: inside it)."""
        return self.lowest_point_z(q) - self._z_min

    def check_command(self, q: np.ndarray) -> None:
        """Reject a commanded pose that brings the arm within the table clearance."""
        z = self.lowest_point_z(q)
        if z < self._z_min:
            raise SafetyAbort(f"command reaches z={z:.3f} m, below table clearance {self._z_min:.3f} m")

    def check_status(self, status: tuple[MotorStatus, ...]) -> None:
        """Abort on an overheating or overloaded motor (NaN readings are ignored)."""
        for motor in status:
            if motor.temperature_c > self._cfg.max_temperature_c:
                raise SafetyAbort(f"{motor.name} temperature {motor.temperature_c:.0f} C over limit")
            if motor.load_fraction > self._cfg.max_load_fraction:
                raise SafetyAbort(f"{motor.name} load {motor.load_fraction:.0%} over limit")

    def record_timing(self, overran: bool) -> None:
        """Count consecutive control-loop overruns; abort at the configured limit."""
        self._consecutive_overruns = self._consecutive_overruns + 1 if overran else 0
        if self._consecutive_overruns >= self._cfg.max_consecutive_overruns:
            raise SafetyAbort(f"{self._consecutive_overruns} consecutive control-loop overruns")

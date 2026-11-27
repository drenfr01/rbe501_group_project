"""Phase 1 mirror tool (and the Phase 3 dry run): the MuJoCo arm follows the real arm.

Connects with calibrate=False and turns torque off, so the arm moves freely by
hand; support it before starting. A 30 Hz loop maps LeRobot readings to sim
angles with the configured `JointMap` and shows them in the passive viewer,
with a table of LeRobot and sim degrees and the lowest monitored point above
the table (what `SafetyMonitor` would see).

Keys:
  r   record the current pose with a reference note (e.g. "upper arm level")
  q   quit

Readings go to data/joint_mapping/<date>_<robot_id>/references.json.
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from rbe_501_so101.config import HardwareConfig, MirrorConfig, SimConfig, resolve_project_path
from rbe_501_so101.envs.model_indices import ModelIndices, table_clearance_z
from rbe_501_so101.hardware.bus import LeRobotJointBus
from rbe_501_so101.hardware.joint_mapping import JointMap, pos_key
from rbe_501_so101.hardware.operator import ABORT_KEY, TerminalKeys, redraw
from rbe_501_so101.hardware.references import ReferenceReading, ReferenceRepository
from rbe_501_so101.hardware.safety import SafetyMonitor
from rbe_501_so101.provenance import begin_run, experiment_dir, require_clean_tree

RECORD_KEY = "r"


class Mirror:
    """Model, viewer state, and mapping for one mirror session."""

    def __init__(self, sim_cfg: SimConfig, hw: HardwareConfig) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(sim_cfg.model_path))
        self.data = mujoco.MjData(self.model)
        self.ids = ModelIndices.from_model(self.model, sim_cfg)
        self.joint_map = JointMap.from_model(self.model, sim_cfg, hw)
        self.safety = SafetyMonitor.from_model(self.model, sim_cfg)
        self.table_z = table_clearance_z(self.model, sim_cfg.table_top_geom, 0.0)

    def show(self, reading: dict[str, float]) -> np.ndarray:
        """Write the mapped pose into the model and return sim radians."""
        q = self.joint_map.to_sim(reading)
        self.data.qpos[self.ids.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return q

    def table(self, reading: dict[str, float], q: np.ndarray, n_saved: int) -> list[str]:
        """Per-joint LeRobot vs sim degrees, plus table clearance."""
        clearance_mm = 1000.0 * (self.safety.lowest_point_z(q) - self.table_z)
        lines = [f"references saved: {n_saved}   clearance {clearance_mm:7.1f} mm   [{RECORD_KEY}] record  [{ABORT_KEY}] quit"]
        lines.append(f"  {'joint':15s} {'lerobot':>9s} {'sim deg':>9s}")
        for name, sim_deg in zip(self.joint_map.joints, np.rad2deg(q), strict=True):
            lines.append(f"  {name:15s} {reading[pos_key(name)]:9.2f} {sim_deg:9.2f}")
        return lines


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command line."""
    hw = HardwareConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=hw.port)
    parser.add_argument("--robot-id", default=hw.robot_id)
    parser.add_argument("--out", type=Path, default=None, help="folder for references.json")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def record(mirror: Mirror, repo: ReferenceRepository, keys: TerminalKeys, reading: dict[str, float], q: np.ndarray) -> None:
    """Ask for a reference note and append the reading."""
    note = keys.prompt("\nreference note: ")
    repo.append(ReferenceReading(note, dict(reading), tuple(float(v) for v in np.rad2deg(q))))


def loop(mirror: Mirror, bus: LeRobotJointBus, repo: ReferenceRepository, period_s: float) -> None:
    """Mirror until the viewer closes or the operator quits."""
    with mujoco.viewer.launch_passive(mirror.model, mirror.data) as viewer, TerminalKeys() as keys:
        first = True
        while viewer.is_running() and (key := keys.poll(0.0)) != ABORT_KEY:
            started = time.monotonic()
            reading = bus.read_positions()
            q = mirror.show(reading)
            viewer.sync()
            if key == RECORD_KEY:
                record(mirror, repo, keys, reading, q)
                first = True
            redraw(mirror.table(reading, q, len(repo.load())), first)
            first = False
            time.sleep(max(0.0, period_s - (time.monotonic() - started)))


def main(argv: list[str] | None = None) -> None:
    """Entry point; torque stays off for the whole session."""
    args = parse_args(argv)
    require_clean_tree(args.allow_dirty)
    hw, cfg = replace(HardwareConfig(), port=args.port, robot_id=args.robot_id), MirrorConfig()
    out = resolve_project_path(args.out) if args.out else experiment_dir(hw.robot_id, base=cfg.references_dir)
    manifest = begin_run(out, hw.robot_id)
    bus = LeRobotJointBus.connect(hw, calibrate=False)
    bus.set_torque(False)
    try:
        loop(Mirror(SimConfig(), hw), bus, ReferenceRepository(out / cfg.references_filename), 1.0 / cfg.loop_hz)
    finally:
        bus.close()
    manifest.finalize(out)


if __name__ == "__main__":
    main()

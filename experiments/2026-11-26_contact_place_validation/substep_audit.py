"""Read-only physics-step checks of selected repaired and fresh complete sequences."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from rbe_501_so101.control.contact_tracked_place import ContactTrackedCarryPlace
from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.pick_lift_coverage import DiagnosticPlacer, PoseCase

ROOT = Path(__file__).parent


class AuditedSequence(ContactTrackedCarryPlace):
    """Record pre-opening physical support without changing any guard or command."""

    def release(self) -> bool:
        """Measure the cube/table normal force before the inherited release decision."""
        self.pre_release_ready = self.support_ready()
        self.pre_release_force_n = 0.
        model, data = self.backend.model, self.backend.data
        wrench = np.zeros(6)
        for i, contact in enumerate(data.contact):
            pair = {int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
            if self.table in pair and self.cube in pair:
                mujoco.mj_contactForce(model, data, i, wrench)
                self.pre_release_force_n += float(wrench[0])
        return super().release()


def episode(repo, seed: int, mass: float, yaw: float) -> dict:
    """Restore mj_step after each episode, including failed controller calls."""
    preset = repo.load()
    preset = replace(preset, base=replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass)))
    env = make_timed_env(preset)
    raw = env.unwrapped
    raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z, PoseCase("audit", yaw_offset_deg=yaw))
    sequence = AuditedSequence(env)
    original = mujoco.mj_step
    result = {"seed": seed, "mass_kg": mass, "yaw_offset_deg": yaw, "physics_steps": 0, "robot_table_contact_steps": 0}

    def checked(model, data):
        """Read actual contacts after each physical advance; do not alter simulation state."""
        original(model, data)
        if model is sequence.backend.model and data is sequence.backend.data:
            result["physics_steps"] += 1
            result["robot_table_contact_steps"] += int(sequence.contacts()[1])

    mujoco.mj_step = checked
    try:
        success = sequence.acquire(seed) and sequence.run_transfer()
        result.update(success=bool(success), failure=sequence.failure,
                      support_ready_before_open=getattr(sequence, "pre_release_ready", False),
                      cube_table_force_before_open_n=getattr(sequence, "pre_release_force_n", None),
                      warnings=sequence.backend.has_warnings())
    finally:
        mujoco.mj_step = original
        env.close()
    return result


def run() -> None:
    """Audit the last timeout, the earlier tracking regression, and two fresh conditions."""
    repo = CartesianZeroRepository(Path("checkpoints/calibrated_cartesian_zero_v3_pilot"))
    conditions = ((89000, .005, 30), (88000, .015, -30), (91000, .005, -30), (91000, .02, 30))
    rows = [episode(repo, *condition) for condition in conditions]
    write(ROOT / "substep_audit.json", rows)
    print("Audit", sum(r["success"] for r in rows), "/", len(rows),
          "robot/table", sum(r["robot_table_contact_steps"] for r in rows), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()

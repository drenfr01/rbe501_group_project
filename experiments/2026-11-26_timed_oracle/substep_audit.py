"""Read-only contact instrumentation at every 1 ms physics step in selected cases."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
from rbe_501_so101.envs.model_indices import descendant_body_ids
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.pick_lift_coverage import DiagnosticPlacer, PoseCase

ROOT = Path(__file__).parent


def audit(env, case: PoseCase, seed: int) -> dict:
    """Wrap only this process's mj_step; restore it even if the rollout fails."""
    raw = env.unwrapped
    raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z, case)
    env.reset(seed=seed)
    model, data = raw.backend.model, raw.backend.data
    robot = descendant_body_ids(model, raw.config.sim.robot_root_body)
    table, cube = model.body(raw.config.sim.table_body).id, model.body(raw.config.object.body).id
    initial = data.xpos[cube, :2].copy()
    result = {"case": case.name, "seed": seed, "mass_kg": raw.backend.object_mass_kg,
              "substeps": 0, "table_contact_substeps": 0, "approach_cube_contact_substeps": 0,
              "preclose_max_xy_mm": 0.}
    original = mujoco.mj_step

    def checked(current_model, current_data):
        """Measure contacts after advancing physics, without changing commands or state."""
        original(current_model, current_data)
        if current_model is not model or current_data is not data:
            return
        table_hit = cube_hit = False
        for contact in data.contact:
            bodies = {int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
            table_hit |= table in bodies and bool(robot & bodies)
            cube_hit |= cube in bodies and bool(robot & bodies)
        result["substeps"] += 1
        result["table_contact_substeps"] += int(table_hit)
        result["approach_cube_contact_substeps"] += int(env._controller._step <= 40 and cube_hit)
        if not env._controller.close_started:
            shift = float(np.linalg.norm(data.xpos[cube, :2] - initial) * 1000)
            result["preclose_max_xy_mm"] = max(result["preclose_max_xy_mm"], shift)

    mujoco.mj_step = checked
    try:
        for _ in range(raw.config.task.max_episode_steps):
            _, _, _, _, info = env.step(np.zeros(6))
    finally:
        mujoco.mj_step = original
    result.update(success=bool(info["is_success"]), warning=raw.backend.has_warnings())
    return result


def run() -> None:
    """Audit both recorded repairs and two independent 45-degree conditions at all masses."""
    repo = CartesianZeroRepository(Path("checkpoints/calibrated_cartesian_zero_v3_pilot"))
    preset, rows = repo.load(), []
    probes = ((PoseCase("grid", .20, -40), 83000), (PoseCase("recorded_minus30", yaw_offset_deg=-30), 85007),
              (PoseCase("fresh_minus45", yaw_offset_deg=-45), 86000), (PoseCase("fresh_plus45", yaw_offset_deg=45), 86000))
    for mass in (.005, .01, .015, .02):
        cfg = replace(preset, base=replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass)))
        for case, seed in probes:
            env = make_timed_env(cfg)
            try:
                rows.append(audit(env, case, seed))
            finally:
                env.close()
    write(ROOT / "substep_audit.json", rows)
    print("Substep audit", sum(r["success"] for r in rows), "/", len(rows),
          "table", sum(r["table_contact_substeps"] for r in rows), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()

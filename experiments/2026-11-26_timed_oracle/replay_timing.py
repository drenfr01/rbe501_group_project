"""Replay repaired cases and record actual phase, readiness and contact-release timing."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.pick_lift_coverage import DiagnosticPlacer, PoseCase
from scripts.sim2real.analyze.replay_pick_lift import FRAME_STEPS, save_frame, snapshot

ROOT = Path(__file__).parent


def trace(env, seed: int, output: Path) -> list[dict]:
    """Record real acquisition state rather than relying on nominal clock phase labels."""
    env.reset(seed=seed)
    raw, rows = env.unwrapped, []
    renderer = mujoco.Renderer(raw.backend.model, 480, 640)
    try:
        for step in range(raw.config.task.max_episode_steps):
            _, _, _, _, info = env.step(np.zeros(6))
            controller = env._controller
            rows.append({**snapshot(raw, info, step, np.zeros(6)), "actual_phase": str(controller.phase),
                "stage": controller.descent_stage, "close_started": controller.close_started,
                "close_start_step": controller.close_start_step, "stable_count": controller.alignment_count,
                "abort_reason": controller.abort_reason, "real_step": controller._step})
            if step in FRAME_STEPS:
                save_frame(renderer, raw, output / f"step_{step + 1:03d}_{controller.phase}.png")
    finally:
        renderer.close()
    return rows


def run() -> None:
    """Keep the mass and recorded placements fixed for direct regression reproduction."""
    repo = CartesianZeroRepository(Path("checkpoints/calibrated_cartesian_zero_v3_pilot"))
    preset = repo.load()
    preset = replace(preset, base=replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=.01)))
    for case, seed in ((PoseCase("grid_repaired", .20, -40), 83000),
                       (PoseCase("yaw_repaired", yaw_offset_deg=-30), 85007)):
        output = ROOT / case.name
        output.mkdir(parents=True, exist_ok=False)
        env = make_timed_env(preset)
        raw = env.unwrapped
        raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z, case)
        try:
            rows = trace(env, seed, output)
            write(output / "trace.json", rows)
            print(case.name, "close", next(r["real_step"] for r in rows if r["close_started"]),
                  "lift", next(r["real_step"] for r in rows if r["lift_released"]),
                  "success", rows[-1]["success"], flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()

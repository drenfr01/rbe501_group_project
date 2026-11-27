"""Replay one repaired rotation and one new nominal-grid regression, without tuning."""

from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.control.oracle_acquisition import make_oracle_env
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository
from scripts.sim2real.analyze.calibrated_lift_pilot import ZeroResidual, write
from scripts.sim2real.analyze.pick_lift_coverage import DiagnosticPlacer, PoseCase
from scripts.sim2real.analyze.replay_pick_lift import rollout

ROOT = Path(__file__).parent


def run() -> None:
    """Retain actual contact/image evidence and the final measured closure errors."""
    repo = CartesianZeroRepository(Path("checkpoints/calibrated_cartesian_zero_v3_pilot"))
    probes = ((PoseCase("repaired_yaw45", yaw_offset_deg=45), 82000),
              (PoseCase("nominal_grid_regression", .20, -40), 83000))
    for case, seed in probes:
        output = ROOT / case.name
        output.mkdir(parents=True, exist_ok=False)
        env = make_oracle_env(repo.load())
        raw = env.unwrapped
        raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z, case)
        try:
            trace = rollout(env, ZeroResidual(), seed, output, frames=True, viewer=False)
            controller = env._controller
            write(output / "trace.json", {"trace": trace, "mass_kg": raw.backend.object_mass_kg,
                "seed": seed, "close_started": controller.close_started, "abort_reason": controller.abort_reason,
                "measured_closure_error_xyz_mm": (controller.alignment_error * 1000).tolist(),
                "measured_jaw_error_deg": float(np.rad2deg(controller.jaw_alignment_rad))})
            print(case.name, trace[-1]["success"], controller.alignment_error * 1000, flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()

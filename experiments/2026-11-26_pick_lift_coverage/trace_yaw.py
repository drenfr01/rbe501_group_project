"""Reproduce one nominal and two rotated acquisitions with contact traces/images."""

from pathlib import Path

import torch

from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository, make_env
from scripts.sim2real.analyze.calibrated_lift_pilot import ZeroResidual, write
from scripts.sim2real.analyze.pick_lift_coverage import DiagnosticPlacer, PoseCase
from scripts.sim2real.analyze.replay_pick_lift import rollout, transition_events

ROOT = Path(__file__).parent


def run() -> None:
    """Leave both saved controllers unchanged; use the scripted acquisition for diagnosis."""
    repo = CartesianZeroRepository(Path("checkpoints/calibrated_cartesian_zero_v3_pilot"))
    for yaw in (0, 30, 45):
        output = ROOT / f"trace_yaw{yaw}_seed82000"
        output.mkdir(parents=True, exist_ok=False)
        env = make_env(repo.load())
        raw = env.unwrapped
        raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z,
                                      PoseCase(f"yaw{yaw}", yaw_offset_deg=yaw))
        try:
            trace = rollout(env, ZeroResidual(), 82000, output, frames=True, viewer=False)
            write(output / "trace.json", {"seed": 82000, "mass_kg": raw.backend.object_mass_kg,
                  "yaw_offset_deg": yaw, "events": transition_events(trace), "trace": trace,
                  "abort_reason": env._controller.abort_reason})
            print(yaw, "success", trace[-1]["success"], "abort", env._controller.abort_reason)
        finally:
            env.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()

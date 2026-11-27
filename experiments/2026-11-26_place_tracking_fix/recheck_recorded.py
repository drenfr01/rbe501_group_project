"""Recheck both prior supported-transfer batches with one frozen lowering correction."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from rbe_501_so101.control.carry_place import TransferConfig
from rbe_501_so101.control.tracked_place import PlacementTrackingConfig, TrackedCarryPlace
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.carry_place_check import episode
from scripts.sim2real.analyze.cartesian_disturbance import fingerprints

ROOT = Path(__file__).parent


def run() -> None:
    """Include every previously successful condition, not just the one recorded failure."""
    repo = CartesianZeroRepository(Path("checkpoints/calibrated_cartesian_zero_v3_pilot"))
    source = Path("experiments/2026-11-26_supported_place")
    previous = json.loads((source / "recorded/episodes.json").read_text()) + json.loads((source / "episodes.json").read_text())
    hashes = fingerprints(repo)
    output = ROOT / "recorded"
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", {"conditions": len(previous), "transfer": asdict(TransferConfig()),
          "tracking": asdict(PlacementTrackingConfig()), "input_sha256": hashes, "source": str(source)})
    rows = []
    for n, case in enumerate(previous, 1):
        rows.append(episode(repo, TransferConfig(), case["seed"], case["mass_kg"], case["yaw_offset_deg"],
                            sequence_type=TrackedCarryPlace))
        if n % 30 == 0:
            write(output / "episodes.json", rows)
            print("Recorded", n, "complete", sum(r["complete_success"] for r in rows), flush=True)
    write(output / "assessment.json", {"episodes": len(rows), "successes": sum(r["complete_success"] for r in rows),
          "previous_successes": sum(r["complete_success"] for r in previous),
          "new_failures": sum(a["complete_success"] and not b["complete_success"] for a, b in zip(previous, rows)),
          "table_contacts": sum(r["table_contact_steps"] for r in rows), "warnings": sum(r["warning_steps"] for r in rows)})
    assert hashes == fingerprints(repo)
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()

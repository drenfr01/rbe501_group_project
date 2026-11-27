"""Recheck all 240 recorded complete-task conditions, including both tracking stages."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from rbe_501_so101.control.carry_place import TransferConfig
from rbe_501_so101.control.contact_tracked_place import ContactTrackedCarryPlace
from rbe_501_so101.control.tracked_place import PlacementTrackingConfig
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.carry_place_check import episode
from scripts.sim2real.analyze.cartesian_disturbance import fingerprints

ROOT = Path(__file__).parent


def run() -> None:
    """Preserve every previous success, using all recorded conditions rather than selected repairs."""
    repo = CartesianZeroRepository(Path("checkpoints/calibrated_cartesian_zero_v3_pilot"))
    sources = [Path("experiments/2026-11-26_supported_place/recorded/episodes.json"),
               Path("experiments/2026-11-26_supported_place/episodes.json"),
               Path("experiments/2026-11-26_place_tracking_validation/episodes.json")]
    previous = [r for source in sources for r in json.loads(source.read_text())]
    hashes = fingerprints(repo)
    output = ROOT / "recorded"
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", {"sources": [str(p) for p in sources], "conditions": len(previous),
          "transfer": asdict(TransferConfig()), "tracking": asdict(PlacementTrackingConfig()), "input_sha256": hashes,
          "contact_rule": "No vertical integration on table support; hold current vertical reference during supported lateral correction."})
    rows = []
    for n, case in enumerate(previous, 1):
        rows.append(episode(repo, TransferConfig(), case["seed"], case["mass_kg"], case["yaw_offset_deg"],
                            sequence_type=ContactTrackedCarryPlace))
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

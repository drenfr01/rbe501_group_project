"""Recheck every first-prototype condition with the frozen support probe, without selecting cases."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from rbe_501_so101.control.carry_place import TransferConfig
from rbe_501_so101.control.supported_place import SupportSearchConfig, SupportedCarryPlace
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.carry_place_check import episode
from scripts.sim2real.analyze.cartesian_disturbance import fingerprints

ROOT = Path(__file__).parent


def run() -> None:
    """Preserve the 34/60 trial and check both its successes and its failures unchanged."""
    repo = CartesianZeroRepository(Path("checkpoints/calibrated_cartesian_zero_v3_pilot"))
    source = Path("experiments/2026-11-26_carry_place")
    cases = json.loads((source / "episodes.json").read_text())
    hashes = fingerprints(repo)
    output = ROOT / "recorded"
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", {"source": str(source), "conditions": len(cases),
          "transfer": asdict(TransferConfig()), "support": asdict(SupportSearchConfig()), "input_sha256": hashes})
    rows = []
    for case in cases:
        rows.append(episode(repo, TransferConfig(), case["seed"], case["mass_kg"], case["yaw_offset_deg"],
                            sequence_type=SupportedCarryPlace))
    write(output / "episodes.json", rows)
    write(output / "assessment.json", {"episodes": len(rows), "successes": sum(r["complete_success"] for r in rows),
          "new_failures": sum(a["complete_success"] and not b["complete_success"] for a, b in zip(cases, rows)),
          "table_contacts": sum(r["table_contact_steps"] for r in rows),
          "warnings": sum(r["warning_steps"] for r in rows)})
    assert hashes == fingerprints(repo)
    write(output / "input_verification.json", {"unchanged": True})
    print("Recorded complete", sum(r["complete_success"] for r in rows), "/", len(rows), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()

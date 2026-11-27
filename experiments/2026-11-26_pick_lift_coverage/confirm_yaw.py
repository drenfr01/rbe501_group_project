"""Independent placement replication of the same fixed yaw ladder; no tuning."""

from dataclasses import asdict, replace
from pathlib import Path

import torch

from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.pick_lift_coverage import CoverageProtocol, fingerprints, summarize, sweep

ROOT = Path(__file__).parent / "confirmation"


def run() -> None:
    """Repeat all seven yaw offsets on ten new placements without retuning either controller."""
    ROOT.mkdir(parents=True, exist_ok=False)
    repo = CartesianZeroRepository(Path("checkpoints/calibrated_cartesian_zero_v3_pilot"))
    protocol = replace(CoverageProtocol(), radii_m=(), first_yaw_seed=84000)
    hashes = fingerprints(repo)
    write(ROOT / "protocol.json", {"protocol": asdict(protocol), "input_sha256": hashes,
          "purpose": "Independent replication of the entire original yaw ladder; no selection of angles or controller changes.",
          "decision": "Shared acquisition failures plus absent pre-contact orientation observations rule out lift-only retraining as a fix."})
    rows = sweep(repo, ROOT, protocol)
    write(ROOT / "summary.json", summarize(rows))
    assert hashes == fingerprints(repo)
    write(ROOT / "input_verification.json", {"unchanged": True, "sha256": hashes})


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()

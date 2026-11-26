"""Freeze a wrist-seed ablation before testing fresh placements; never train."""

import argparse
import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.control.aligned_pick_lift import MotionOptions
from scripts.sim2real.analyze.pick_lift_calibration import input_hashes
from scripts.sim2real.analyze.pick_lift_motion_ablation import VARIANTS, choose, evaluate, metrics, write, motion_options


def assess(records: list[dict]) -> dict:
    """Require 95% success and <=5% sampled table contact in every mass/mode."""
    summaries = [record["summary"] for record in records]
    passed = all(s["success_rate"] >= .95 and
                 s["robot_table_contact_episodes"] / s["episodes"] <= .05 for s in summaries)
    return {"passed_heldout_gate": passed, "ready_for_training": False,
            "promote_defaults": False, "metrics": metrics(records),
            "reason": "Separate explicit training preset and short pilot still required."}


def run(output: Path) -> None:
    """Tune three seeds on 62000–62009, then freeze before testing 63000–63049."""
    hashes = input_hashes()
    output.mkdir(parents=True, exist_ok=False)
    base = VARIANTS[4]
    options = (base, replace(base, name="opposite_wrist", wrist_seed_offset_rad=-np.pi),
               replace(base, name="positive_wrist", wrist_seed_offset_rad=np.pi))
    records = []
    for option in options:
        results = [record for effects in (False, True)
                   for record in evaluate(option, effects, "scripted", 10, 62000)]
        records.append({"options": asdict(option), "metrics": metrics(results), "results": results})
        write(output / "tuning.json", records)
    selection = {**choose(records), "input_sha256": hashes,
                 "tuning_seeds": [62000, 62009], "heldout_seeds": [63000, 63049]}
    write(output / "frozen_selection.json", selection)
    heldout = []
    for option in (base, MotionOptions(**selection["options"])):
        for effects in (False, True):
            heldout += evaluate(option, effects, "scripted", 50, 63000)
            write(output / "heldout.json", heldout)
    candidate = [r for r in heldout if r["summary"]["dynamics"] == selection["options"]["name"]]
    write(output / "assessment.json", assess(candidate))
    assert hashes == input_hashes(), "Original evaluation inputs changed"


def manifest() -> dict:
    """Identify source files without modifying any existing experiment record."""
    paths = (Path(__file__), Path("src/rbe_501_so101/control/aligned_pick_lift.py"))
    return {"source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "note": "Old failed placements used only for diagnosis; fresh tune and held-out seeds."}


def run_adaptive(output: Path) -> None:
    """Freeze the outer-10-mm workspace rule before testing seeds 65000–65049."""
    hashes = input_hashes()
    output.mkdir(parents=True, exist_ok=False)
    candidate = motion_options("outer_wrist")
    write(output / "frozen_selection.json", {"options": asdict(candidate), "input_sha256": hashes,
          "heldout_seeds": [65000, 65049], "criterion": "Each mass/mode >=95% success and <=5% table contact"})
    records = []
    for option in (VARIANTS[4], candidate):
        for effects in (False, True):
            records += evaluate(option, effects, "scripted", 50, 65000)
            write(output / "heldout.json", records)
    selected = [r for r in records if r["summary"]["dynamics"] == candidate.name]
    write(output / "assessment.json", assess(selected))
    assert hashes == input_hashes(), "Original evaluation inputs changed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adaptive", action="store_true", help="Evaluate the frozen outer-reach wrist rule")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = manifest()
    (run_adaptive if args.adaptive else run)(args.out)
    write(args.out / "manifest.json", source)

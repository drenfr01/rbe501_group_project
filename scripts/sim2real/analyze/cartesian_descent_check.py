"""Freeze Cartesian descent ablations before new paired headless evaluation."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from rbe_501_so101.config import PickLiftEvalConfig
from rbe_501_so101.control.cartesian_descent import CartesianDescent, CartesianGuard, DescentConfig
from rbe_501_so101.control.raised_approach import ApproachConfig
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.limited_lift_pilot import protected_inputs
from scripts.sim2real.analyze.pick_lift_calibration import make_config
from scripts.sim2real.analyze.raised_approach_check import build as raised_build, episode, metrics, passes


@dataclass(frozen=True)
class Candidate:
    """Explicitly distinguish existing references from a new Cartesian controller."""

    name: str
    descent: DescentConfig | None = None
    raised: bool = False


BASELINE = Candidate("baseline")
RAISED = Candidate("raise_only", raised=True)
VARIANTS = (BASELINE, RAISED,
            Candidate("cartesian_no_feedback", DescentConfig(name="cartesian_no_feedback", feedback=False)),
            Candidate("cartesian_table", DescentConfig(name="cartesian_table", check_cube=False)),
            Candidate("cartesian_checked", DescentConfig()))


def build(candidate: Candidate, mass: float, effects: bool, render_mode=None):
    """Keep model, gains and the 200-step horizon; do not execute any trained policy."""
    if candidate.descent is None:
        cfg = ApproachConfig(candidate.name, raise_first=candidate.raised, alignment_gate=False)
        return raised_build(cfg, mass, effects, render_mode)
    raw = SO101PickLiftEnv(cfg=make_config(mass, True), render_mode=render_mode)
    policy = CartesianDescent.for_env(raw, candidate.descent)
    if effects:
        raw.backend = GraspServoBackend(raw.backend, raw.config)
    return CartesianGuard(raw, policy), policy


def evaluate(candidate: Candidate, seeds: range | tuple[int, ...]) -> list[dict]:
    """Retain aborts as failures and record sampled-clearance rejection diagnostics."""
    rows = []
    for effects in (False, True):
        for mass in PickLiftEvalConfig().masses_kg:
            env, policy = build(candidate, mass, effects)
            try:
                episodes = []
                for seed in seeds:
                    result = asdict(episode(env, policy, seed))
                    if candidate.descent is not None:
                        result.update(path_stage=policy.descent_stage, clearance_rejections=policy.clearance_rejections,
                                      closure_rejections=policy.closure_rejections, close_start_step=policy.close_start_step)
                    episodes.append(result)
                rows.append({"variant": candidate.name, "arm_effects": effects, "mass_kg": mass, "episodes": episodes})
                print(candidate.name, effects, mass, "success", sum(e["success"] for e in episodes), "/", len(episodes),
                      "table", sum(e["table_steps"] > 0 for e in episodes), flush=True)
            finally:
                env.close()
    return rows


def choose(records: list[dict]) -> dict:
    """Compare the best Cartesian prototype even if no candidate qualifies for promotion."""
    candidates = [r for r in records if r["candidate"]["descent"] is not None]
    eligible = [r for r in candidates if passes(r["metrics"]) and
                all(e["success"] for row in r["known"] for e in row["episodes"])]
    best = max(eligible or candidates, key=lambda r: (r["metrics"]["worst_success"],
               -r["metrics"]["table_episodes"], -r["metrics"]["worst_shift_rate"]))
    return {"candidate": best["candidate"], "passed_tuning_gate": bool(eligible), "metrics": best["metrics"]}


def decode(value: dict) -> Candidate:
    """Reconstruct the frozen experimental parameters for held-out tests and replay."""
    descent = None if value["descent"] is None else DescentConfig(**value["descent"])
    return Candidate(value["name"], descent, value["raised"])


def run(output: Path) -> None:
    """Use 74000-74004 tuning, then freeze before 75000-75019 independent evaluation."""
    output.mkdir(parents=True, exist_ok=False)
    hashes, records = protected_inputs(), []
    write(output / "protocol.json", {"variants": [asdict(c) for c in VARIANTS], "protected_sha256": hashes,
          "known_seeds": [68000, 68006, 69005], "tuning_seeds": [74000, 74004], "heldout_seeds": [75000, 75019],
          "gate": ">=95% each mass/mode; zero table/approach contact; <=5% episodes >1mm preclose XY displacement; known regressions must lift"})
    for candidate in VARIANTS:
        known = evaluate(candidate, (68000, 68006, 69005))
        rows = evaluate(candidate, range(74000, 74005))
        records.append({"candidate": asdict(candidate), "known": known, "rows": rows, "metrics": metrics(rows)})
        write(output / "tuning.json", records)
    selection = choose(records)
    write(output / "frozen_selection.json", selection)
    heldout = []
    for candidate in (BASELINE, RAISED, decode(selection["candidate"])):
        heldout += evaluate(candidate, range(75000, 75020))
        write(output / "heldout.json", heldout)
    chosen = [r for r in heldout if r["variant"] == selection["candidate"]["name"]]
    result = metrics(chosen)
    write(output / "assessment.json", {"metrics": result, "passed_heldout_gate": passes(result),
                                      "passed_tuning_gate": selection["passed_tuning_gate"],
                                      "overall_validated": bool(selection["passed_tuning_gate"] and passes(result)),
                                      "promote_defaults": False, "train": False})
    assert hashes == protected_inputs(), "Original input files changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


def replay(output: Path, candidate: Candidate, mass: float, seed: int, viewer: bool) -> None:
    """Render the frozen prototype with arm effects; keep all original defaults intact."""
    from scripts.sim2real.analyze.replay_pick_lift import rollout, transition_events

    output.mkdir(parents=True, exist_ok=False)
    env, policy = build(candidate, mass, True, "human" if viewer else None)
    try:
        trace = rollout(env, policy, seed, output, frames=True, viewer=viewer)
        write(output / "trace.json", {"candidate": asdict(candidate), "mass_kg": mass, "seed": seed,
              "events": transition_events(trace), "trace": trace, "abort_reason": policy.abort_reason,
              "path_stage": getattr(policy, "descent_stage", None)})
        print("Replay final success", trace[-1]["success"], "abort", policy.abort_reason, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--mass-kg", type=float, default=.005)
    parser.add_argument("--placement-seed", type=int, default=68006)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.replay:
        replay(args.out, decode(json.loads(args.replay.read_text())["candidate"]), args.mass_kg, args.placement_seed, args.viewer)
    else:
        run(args.out)

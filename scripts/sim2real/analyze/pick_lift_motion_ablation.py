"""Scripted acquisition/lift ablations with unchanged gains, contact physics and policies.

    uv run --locked python -m scripts.sim2real.analyze.pick_lift_motion_ablation --out experiments/pick_lift_motion_ablation

Tuning: 60000-60009; held-out: 61000-61019. Square-face yaw uses the current
TablePlacer's radial-yaw rule, not general object-pose perception. No training.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.config import PickLiftEvalConfig, TrainConfig
from rbe_501_so101.control.aligned_pick_lift import AlignedPickLift, MotionOptions
from rbe_501_so101.control.guarded_pick_lift import GuardedLift
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from rbe_501_so101.rl.pick_lift import load_policy, load_run
from rbe_501_so101.rl.residual import ResidualPickLift
from scripts.sim2real.analyze.pick_lift_calibration import CONTROLLERS, evaluate_episode, input_hashes, make_config, summarize

VARIANTS = (
    MotionOptions("baseline"),
    MotionOptions("gravity_only", gravity_feedforward=True),
    MotionOptions("yaw_only", radial_jaw_alignment=True),
    MotionOptions("gravity_yaw", gravity_feedforward=True, radial_jaw_alignment=True),
    MotionOptions("gravity_yaw_depth4", gravity_feedforward=True, radial_jaw_alignment=True, grasp_height_m=-.004),
    MotionOptions("upright_lift_only", upright_lift=True),
    MotionOptions("gravity_yaw_depth4_upright", gravity_feedforward=True, radial_jaw_alignment=True,
                  grasp_height_m=-.004, upright_lift=True),
)


def motion_options(name: str) -> MotionOptions:
    """Resolve an explicit experiment name without changing the original sweep."""
    if name == "outer_wrist":
        return replace(VARIANTS[4], name=name, opposite_wrist_radius_m=.21)
    return next(option for option in VARIANTS if option.name == name)


def build(options: MotionOptions, effects: bool, mass: float, controller: str, render_mode=None):
    """Opt into a scripted base variant; keep learned weights and normalizers frozen."""
    raw = SO101PickLiftEnv(cfg=make_config(mass, True), render_mode=render_mode)
    script = AlignedPickLift.for_env(raw, options)
    if effects:
        raw.backend = GraspServoBackend(raw.backend, raw.config)
    env = GuardedLift(raw)
    path = CONTROLLERS[controller]
    if path is None:
        return env, script, script
    run = load_run(path)
    if run.control != "residual" or run.object != raw.config.object:
        env.close()
        raise ValueError("Saved policy task does not match this evaluation")
    return ResidualPickLift(env, script), load_policy(path / "seed_0", run, TrainConfig()), script


def evaluate(options: MotionOptions, effects: bool, controller: str, count: int, seed: int) -> list[dict]:
    """Use paired placements across masses and record best-effort IK misses."""
    results = []
    for mass in PickLiftEvalConfig().masses_kg:
        env, policy, script = build(options, effects, mass, controller)
        try:
            episodes, misses = [], []
            for offset in range(count):
                episodes.append(evaluate_episode(env, policy, seed + offset))
                misses.append(script.ik_misses)
            summary = summarize(options.name, controller, mass, episodes)
            results.append({"summary": asdict(summary), "arm_effects": effects,
                            "episodes": [asdict(e) for e in episodes], "ik_misses": misses})
            print(f"{options.name} effects={effects} {controller} {mass*1000:g}g: {summary.success_rate:.0%}, table={summary.robot_table_contact_episodes}/{count}", flush=True)
        finally:
            env.close()
            getattr(policy, "close", lambda: None)()
    return results


def metrics(results: list[dict]) -> dict:
    """Worst dynamics-condition success/contact rate across the four masses."""
    values = []
    for effects in (False, True):
        episodes = [e for r in results if r["arm_effects"] == effects for e in r["episodes"]]
        values.append((float(np.mean([e["final_success"] for e in episodes])),
                       float(np.mean([e["robot_table_contact_steps"] > 0 for e in episodes]))))
    return {"worst_success": min(v[0] for v in values), "worst_table_rate": max(v[1] for v in values)}


def choose(records: list[dict]) -> dict:
    """Prefer >=95% success and <=5% table contact; otherwise label exploratory."""
    eligible = [r for r in records if r["metrics"]["worst_success"] >= .95 and r["metrics"]["worst_table_rate"] <= .05]
    contact_clear = [r for r in records if r["metrics"]["worst_table_rate"] <= .05]
    pool = eligible or contact_clear or records
    best = max(pool, key=lambda r: (r["metrics"]["worst_success"], -r["metrics"]["worst_table_rate"]))
    return {"options": best["options"], "passed_tuning_gate": bool(eligible), "metrics": best["metrics"]}


def write(path: Path, value) -> None:
    """Save new experiment records; never overwrite pre-existing experiments."""
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run(output: Path) -> None:
    """Freeze scripted selection before evaluating any learned policy on new targets."""
    hashes = input_hashes()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for options in VARIANTS:
        results = []
        for effects in (False, True):
            results += evaluate(options, effects, "scripted", 10, 60000)
        records.append({"options": asdict(options), "metrics": metrics(results), "results": results})
        write(output / "tuning.json", records)
    selection = {**choose(records), "input_sha256": hashes, "tuning_seeds": [60000, 60009], "heldout_seeds": [61000, 61019]}
    write(output / "frozen_selection.json", selection)
    candidate = MotionOptions(**selection["options"])
    heldout = []
    for options in (MotionOptions("baseline"), candidate):
        for effects in (False, True):
            for controller in CONTROLLERS:
                heldout += evaluate(options, effects, controller, 20, 61000)
                write(output / "heldout.json", heldout)
    assert hashes == input_hashes(), "An evaluation input changed"
    print("Motion evaluation complete; original inputs unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New experiment output directory")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.out)

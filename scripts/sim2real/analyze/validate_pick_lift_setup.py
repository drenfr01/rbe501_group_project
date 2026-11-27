"""Separate grasp gating, clearance, and arm-effects validation; never train.

    uv run --locked python -m scripts.sim2real.analyze.validate_pick_lift_setup --out experiments/pick_lift_setup_validation

Tune scripted clearance on seeds 40000-40009, freeze its selection before
testing policies on 50000-50019, and keep baseline/checkpoints unchanged.
"""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.config import PickLiftEvalConfig, ScriptedPickLiftConfig, TrainConfig
from rbe_501_so101.control import ScriptedPickLift
from rbe_501_so101.control.guarded_pick_lift import GuardedLift
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from rbe_501_so101.rl.pick_lift import load_policy, load_run
from rbe_501_so101.rl.residual import ResidualPickLift
from scripts.sim2real.analyze.pick_lift_calibration import (
    CONTROLLERS, evaluate_episode, input_hashes, make_config, summarize,
)


@dataclass(frozen=True)
class Setup:
    """Explicit experimental settings; no existing preset is overwritten."""

    name: str
    clearance_m: float = 0.0
    gated: bool = False
    arm_effects: bool = False
    stable_contact_steps: int = 5


def build(setup: Setup, controller: str, mass: float, render_mode=None):
    """Reuse existing models/policies with explicitly chosen opt-in modifiers."""
    config = make_config(mass, True)
    raw = SO101PickLiftEnv(cfg=config, render_mode=render_mode)
    script_cfg = replace(ScriptedPickLiftConfig(), grasp_height_m=setup.clearance_m)
    script = ScriptedPickLift.for_env(raw, script_cfg)
    if setup.arm_effects:
        raw.backend = GraspServoBackend(raw.backend, config)
    env = GuardedLift(raw, setup.stable_contact_steps) if setup.gated else raw
    path = CONTROLLERS[controller]
    if path is None:
        return env, script
    run = load_run(path)
    if run.control != "residual" or run.object != config.object:
        env.close()
        raise ValueError("Checkpoint task does not match evaluation")
    env = ResidualPickLift(env, script)
    return env, load_policy(path / "seed_0", run, TrainConfig())


def evaluate(setup: Setup, controller: str, count: int, offset: int) -> list[dict]:
    """Evaluate identical placements at every mass and release all resources."""
    results = []
    for mass in PickLiftEvalConfig().masses_kg:
        env, policy = build(setup, controller, mass)
        try:
            rows = [evaluate_episode(env, policy, offset + i) for i in range(count)]
            summary = summarize(setup.name, controller, mass, rows)
            results.append({"summary": asdict(summary), "episodes": [asdict(r) for r in rows]})
            print(f"{setup.name} {controller} {mass*1000:g} g: {summary.success_rate:.0%}, table {summary.robot_table_contact_episodes}/{count}", flush=True)
        finally:
            env.close()
            getattr(policy, "close", lambda: None)()
    return results


def choose(tuning: list[dict]) -> Setup:
    """Require 95% scripted success if available; then minimize table contact."""
    guarded = [r for r in tuning if r["setup"]["gated"]]
    eligible = [r for r in guarded if r["success_rate"] >= 0.95] or guarded
    best = min(eligible, key=lambda r: (r["table_episodes"], -r["success_rate"], r["setup"]["clearance_m"]))
    return Setup(**{**best["setup"], "name": "guarded_stiffness_candidate"})


def write(path: Path, data: dict | list) -> None:
    """Write only newly created experiment artifacts, never training inputs."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def tune(output: Path) -> Setup:
    """Compare gating and clearance independently using scripted control only."""
    setups = [Setup("baseline"), Setup("gate_only", gated=True)]
    setups += [Setup(f"gate_clearance_{mm}mm", mm / 1000, True) for mm in (2, 4, 6, 8)]
    tuning = []
    for setup in setups:
        results = evaluate(setup, "scripted", 10, 40000)
        episodes = [ep for result in results for ep in result["episodes"]]
        tuning.append({"setup": asdict(setup), "results": results,
                       "success_rate": float(np.mean([ep["final_success"] for ep in episodes])),
                       "table_episodes": sum(ep["robot_table_contact_steps"] > 0 for ep in episodes)})
        write(output / "tuning.json", tuning)
    return choose(tuning)


def run(output: Path) -> None:
    """Freeze before held-out evaluation; retain negative results without tuning them away."""
    hashes = input_hashes()
    output.mkdir(parents=True, exist_ok=False)
    selected = tune(output)
    effects = replace(selected, name="guarded_arm_effects_candidate", arm_effects=True)
    frozen = {"selected": asdict(selected), "arm_effects_stress": asdict(effects), "input_sha256": hashes,
              "tuning_seeds": [40000, 40009], "heldout_seeds": [50000, 50019],
              "scope": "Simulation experiment; not hardware-calibrated manipulation approval",
              "selection": "Among guarded settings with >=95% success if any, minimize table-contact episodes; otherwise exploratory only"}
    write(output / "frozen_candidate.json", frozen)
    results = []
    for setup in (Setup("baseline_stiffness"), selected, effects):
        for controller in CONTROLLERS:
            results += evaluate(setup, controller, 20, 50000)
            write(output / "heldout.json", results)
    if hashes != input_hashes():
        raise RuntimeError("An input artifact changed during validation")
    print("Validation complete; inputs unchanged, no training.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New directory; existing results are never overwritten")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.out)

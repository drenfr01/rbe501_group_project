"""Independent raise/alignment ablations; no training and no baseline promotion."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.config import PickLiftEvalConfig
from rbe_501_so101.control.raised_approach import AcquisitionGuard, ApproachConfig, RaisedApproach
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.envs.model_indices import descendant_body_ids
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from scripts.sim2real.analyze.acquisition_holding_checks import physical_state
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.limited_lift_pilot import protected_inputs
from scripts.sim2real.analyze.pick_lift_calibration import make_config, robot_touches_table

BASELINE = ApproachConfig("baseline", raise_first=False, alignment_gate=False)
VARIANTS = (BASELINE, ApproachConfig("raise_only", alignment_gate=False),
            ApproachConfig("correct_only", raise_first=False, correction_enabled=True),
            ApproachConfig("raise_check"), ApproachConfig("raise_correct", correction_enabled=True))


@dataclass(frozen=True)
class AcquisitionResult:
    """Count premature approach contact, object movement, safe aborts and final lift."""

    seed: int
    success: bool
    lost_success: bool
    table_steps: int
    approach_contact_steps: int
    preclose_max_xy_mm: float
    close_step: int | None
    raise_done_step: int | None
    abort_reason: str | None
    alignment_wait_steps: int
    last_alignment_error_mm: list[float] | None
    bias_mm: list[float]
    warning: bool


def build(cfg: ApproachConfig, mass: float, effects: bool, render_mode=None):
    """Reuse the measured gains, scene, original final waypoints and 200-step horizon."""
    raw = SO101PickLiftEnv(cfg=make_config(mass, True), render_mode=render_mode)
    policy = RaisedApproach.for_env(raw, cfg)
    if effects:
        raw.backend = GraspServoBackend(raw.backend, raw.config)
    return AcquisitionGuard(raw, policy), policy


def touches_cube(raw, bodies: frozenset[int]) -> bool:
    """Any robot/cube contact at the control boundary, not only the pad sites."""
    model = raw.backend.model
    cube = model.body(raw.config.object.body).id
    for c in raw.backend.data.contact:
        pair = {int(model.geom_bodyid[g]) for g in (c.geom1, c.geom2)}
        if cube in pair and pair & bodies:
            return True
    return False


def episode(env, policy: RaisedApproach, seed: int) -> AcquisitionResult:
    """Use measured controller inputs; use true simulator poses for displacement metrics."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    raw = env.unwrapped
    initial = physical_state(raw).obj.p_s.copy()
    bodies = descendant_body_ids(raw.backend.model, raw.config.sim.robot_root_body)
    table = contacts = 0
    shift = 0.
    close_step = raised = None
    ever_success = False
    for step in range(raw.config.task.max_episode_steps):
        previous_stage = policy.stage
        action = policy(obs)
        if previous_stage == "raise" and policy.stage == "traverse":
            raised = step + 1
        closing = not policy.aborted and policy._target.gripper_rad < np.deg2rad(raw.config.gripper.open_deg) - 1e-6
        if closing and close_step is None:
            close_step = step + 1
        obs, reward, _, _, info = env.step(action)
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError("Nonfinite acquisition experiment")
        table += int(robot_touches_table(raw, bodies))
        contacts += int(step < 40 and touches_cube(raw, bodies))
        if close_step is None:
            shift = max(shift, float(np.linalg.norm((physical_state(raw).obj.p_s - initial)[:2]) * 1000))
        ever_success |= bool(info["is_success"])
    error = None if policy.alignment_error is None else (policy.alignment_error * 1000).tolist()
    return AcquisitionResult(seed, bool(info["is_success"]), ever_success and not info["is_success"],
                             table, contacts, shift, close_step, raised, policy.abort_reason,
                             policy.alignment_wait_steps, error, (policy.target_bias * 1000).tolist(),
                             raw.backend.has_warnings())


def evaluate(cfg: ApproachConfig, seeds: range | tuple[int, ...]) -> list[dict]:
    """Pair mass, physics and placements for every acquisition variant."""
    rows = []
    for effects in (False, True):
        for mass in PickLiftEvalConfig().masses_kg:
            env, policy = build(cfg, mass, effects)
            try:
                episodes = [asdict(episode(env, policy, seed)) for seed in seeds]
                rows.append({"variant": cfg.name, "arm_effects": effects, "mass_kg": mass, "episodes": episodes})
                print(cfg.name, effects, mass, "success", sum(e["success"] for e in episodes), "/", len(episodes),
                      "table", sum(e["table_steps"] > 0 for e in episodes),
                      "early", sum(e["approach_contact_steps"] > 0 for e in episodes), flush=True)
            finally:
                env.close()
    return rows


def metrics(rows: list[dict]) -> dict:
    """Require >=95% success, no table/approach contact and <=5% >1 mm pre-close shifts."""
    values = []
    for row in rows:
        es = row["episodes"]
        values.append({"success": float(np.mean([e["success"] for e in es])),
                       "table": sum(e["table_steps"] > 0 for e in es),
                       "approach": sum(e["approach_contact_steps"] > 0 for e in es),
                       "shift_rate": float(np.mean([e["preclose_max_xy_mm"] > 1. for e in es])),
                       "warnings": sum(e["warning"] for e in es)})
    return {"worst_success": min(v["success"] for v in values),
            "table_episodes": sum(v["table"] for v in values),
            "approach_contact_episodes": sum(v["approach"] for v in values),
            "worst_shift_rate": max(v["shift_rate"] for v in values),
            "warnings": sum(v["warnings"] for v in values)}


def passes(value: dict) -> bool:
    """An open abort is a failed lift, not a successful collision-avoidance result."""
    return bool(value["worst_success"] >= .95 and value["table_episodes"] == 0 and
                value["approach_contact_episodes"] == 0 and value["worst_shift_rate"] <= .05 and not value["warnings"])


def select(records: list[dict]) -> dict:
    """Select a raised candidate before independent evaluation, retaining rejected ablations."""
    candidates = [r for r in records if r["config"]["raise_first"]]
    eligible = [r for r in candidates if passes(r["metrics"]) and all(e["success"] for row in r["known"] for e in row["episodes"])]
    pool = eligible or candidates
    best = max(pool, key=lambda r: (r["metrics"]["worst_success"], -r["metrics"]["table_episodes"],
                                   -r["metrics"]["approach_contact_episodes"], -r["metrics"]["worst_shift_rate"]))
    return {"config": best["config"], "passed_tuning_gate": bool(eligible), "metrics": best["metrics"]}


def run(output: Path) -> None:
    """Tune on 70000-70009; freeze before script-only held-out seeds 71000-71049."""
    output.mkdir(parents=True, exist_ok=False)
    hashes, records = protected_inputs(), []
    write(output / "protocol.json", {"configs": [asdict(c) for c in VARIANTS], "tuning_seeds": [70000, 70009],
          "known_seeds": [68000, 68006, 69005], "heldout_seeds": [71000, 71049], "protected_sha256": hashes,
          "gate": ">=95% per mass/mode, zero table/approach contact episodes, <=5% >1mm preclose shifts; known regressions must lift"})
    for cfg in VARIANTS:
        known, rows = evaluate(cfg, (68000, 68006, 69005)), evaluate(cfg, range(70000, 70010))
        records.append({"config": asdict(cfg), "known": known, "rows": rows, "metrics": metrics(rows)})
        write(output / "tuning.json", records)
    selection = select(records)
    write(output / "frozen_selection.json", selection)
    heldout = []
    for cfg in (BASELINE, ApproachConfig(**selection["config"])):
        heldout += evaluate(cfg, range(71000, 71050))
        write(output / "heldout.json", heldout)
    selected = [r for r in heldout if r["variant"] == selection["config"]["name"]]
    result = metrics(selected)
    write(output / "assessment.json", {"metrics": result, "passed_heldout_gate": passes(result),
          "promote_defaults": False, "train": False})
    assert hashes == protected_inputs(), "Original model/calibration/checkpoints changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


def replay(output: Path, cfg: ApproachConfig, mass: float, seed: int, effects: bool, viewer: bool) -> None:
    """Display actual controller phases and acquisition diagnostics with phase PNGs."""
    from scripts.sim2real.analyze.replay_pick_lift import rollout, transition_events

    output.mkdir(parents=True, exist_ok=False)
    env, policy = build(cfg, mass, effects, "human" if viewer else None)
    try:
        trace = rollout(env, policy, seed, output, frames=True, viewer=viewer)
        write(output / "trace.json", {"config": asdict(cfg), "mass_kg": mass, "seed": seed,
              "arm_effects": effects, "events": transition_events(trace), "trace": trace,
              "alignment_wait_steps": policy.alignment_wait_steps, "bias_mm": (policy.target_bias * 1000).tolist(),
              "abort_reason": policy.abort_reason})
        print("Final replay success:", trace[-1]["success"], "abort:", policy.abort_reason, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", type=Path, help="Frozen selection JSON to replay; omit to evaluate")
    parser.add_argument("--mass-kg", type=float, default=.005)
    parser.add_argument("--placement-seed", type=int, default=68006)
    parser.add_argument("--arm-effects", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.replay:
        selection = json.loads(args.replay.read_text())
        replay(args.out, ApproachConfig(**selection["config"]), args.mass_kg, args.placement_seed, args.arm_effects, args.viewer)
    else:
        run(args.out)

"""Isolate jaw closing with held arm commands, then test guarded early lift."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.config import PickLiftEvalConfig
from rbe_501_so101.control.cartesian_descent import CartesianDescent, CartesianGuard, DescentConfig
from rbe_501_so101.control.gripper_closing import ClosingLatch, ClosingProfile, closing_action
from rbe_501_so101.control.scripted_pick_lift import PhaseTarget
from rbe_501_so101.envs.grasp_model import FIXED_PAD_GEOM, MOVING_PAD_GEOM
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.envs.model_indices import descendant_body_ids
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from scripts.sim2real.analyze.acquisition_holding_checks import pad_record, physical_state
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.limited_lift_pilot import protected_inputs
from scripts.sim2real.analyze.pick_lift_calibration import make_config, robot_touches_table
from scripts.sim2real.analyze.replay_pick_lift import contact_records


@dataclass(frozen=True)
class ClosingProtocol:
    """Equal diagnostic stage budgets for every profile; not the old 200-step task."""

    acquisition_steps: int = 100
    flush_steps: int = 2
    closing_steps: int = 100
    lift_steps: int = 100
    max_episode_steps: int = 400
    stable_before_lift: int = 5
    early_lift_steps: int = 20


PROFILES = (ClosingProfile("position_minus10"), ClosingProfile("position_zero", target_deg=0.),
            ClosingProfile("slow_minus10", rate_rad=.025),
            ClosingProfile("contact_stop1", stop_on_contact=True),
            ClosingProfile("contact_stop3", stop_on_contact=True, stable_steps=3))


def build_end_to_end(profile: ClosingProfile, mass: float, effects: bool, render_mode=None):
    """Apply a fixed closing endpoint in the original 200-step Cartesian sequence."""
    if profile.stop_on_contact:
        raise ValueError("Contact-latch profiles require a separate end-to-end controller")
    cfg = make_config(mass, True)
    cfg = replace(cfg, gripper=replace(cfg.gripper, close_deg=profile.target_deg))
    raw = SO101PickLiftEnv(cfg=cfg, render_mode=render_mode)
    nominal = DescentConfig(name="cartesian_table", check_cube=False)
    steps = int(np.ceil(np.deg2rad(cfg.gripper.open_deg - profile.target_deg) / profile.rate_rad)) + 5
    descent = replace(nominal, close_dwell_steps=max(nominal.close_dwell_steps, steps))
    policy = CartesianDescent.for_env(raw, descent)
    policy._cfg = replace(policy._cfg, close_rate_rad=profile.rate_rad)
    if effects:
        raw.backend = GraspServoBackend(raw.backend, raw.config)
    return CartesianGuard(raw, policy), policy


@dataclass
class Prepared:
    """Matched physical initial state after acquisition and draining queued arm commands."""

    obs: np.ndarray
    arm_command: np.ndarray
    cube: np.ndarray
    pinch: np.ndarray
    quaternion: np.ndarray
    digest: str


def build(mass: float, effects: bool):
    """Use the frozen table-checked Cartesian acquisition and unchanged physical model."""
    cfg = make_config(mass, True)
    cfg = replace(cfg, task=replace(cfg.task, max_episode_steps=ClosingProtocol().max_episode_steps))
    raw = SO101PickLiftEnv(cfg=cfg)
    policy = CartesianDescent.for_env(raw, DescentConfig(name="cartesian_table", check_cube=False))
    if effects:
        raw.backend = GraspServoBackend(raw.backend, raw.config)
    return CartesianGuard(raw, policy), policy


def prepare(env, policy, seed: int, cfg: ClosingProtocol) -> Prepared | None:
    """Intercept the first authorized closure before applying any jaw-closing action."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    for _ in range(cfg.acquisition_steps):
        action = policy(obs)
        if policy.close_started:
            break
        if policy.aborted:
            return None
        obs, *_ = env.step(action)
    if not policy.close_started:
        return None
    raw = env.unwrapped
    for _ in range(cfg.flush_steps):
        obs, *_ = raw.step(np.zeros(6))
    state = physical_state(raw)
    command = raw.grasp_state.robot.q_cmd.copy()
    body = raw.backend.model.body(raw.config.object.body).id
    digest = hashlib.sha256(np.concatenate([raw.backend.data.qpos, raw.backend.data.qvel, command]).tobytes()).hexdigest()
    return Prepared(obs, command[:5], state.obj.p_s.copy(), state.p_pinch.copy(),
                    raw.backend.data.xquat[body].copy(), digest)


def frame(raw, stage: str, step: int, prepared: Prepared, bodies: frozenset[int]) -> dict:
    """Measure true cube motion, pad forces and command/physical arm deflection separately."""
    state, contacts = physical_state(raw), contact_records(raw)
    body = raw.backend.model.body(raw.config.object.body).id
    quat = raw.backend.data.xquat[body]
    rotation = float(np.rad2deg(2 * np.arccos(np.clip(abs(np.dot(quat, prepared.quaternion)), 0, 1))))
    q = raw.backend.data.qpos[raw.backend.ids.qpos_adr]
    command = raw.grasp_state.robot.q_cmd
    return {"stage": stage, "stage_step": step + 1, "grasped": state.grasped,
            "cube_m": state.obj.p_s.tolist(), "pinch_m": state.p_pinch.tolist(),
            "cube_shift_xy_mm": float(np.linalg.norm((state.obj.p_s - prepared.cube)[:2]) * 1000),
            "cube_rotation_deg": rotation, "pinch_drift_mm": float(np.linalg.norm(state.p_pinch - prepared.pinch) * 1000),
            "arm_command_drift_rad": float(np.max(np.abs(command[:5] - prepared.arm_command))),
            "jaw_deg": float(np.rad2deg(q[-1])), "jaw_command_deg": float(np.rad2deg(command[-1])),
            "fixed": pad_record(raw, FIXED_PAD_GEOM, contacts),
            "moving": pad_record(raw, MOVING_PAD_GEOM, contacts),
            "table_contact": robot_touches_table(raw, bodies), "lift_mm": state.obj.lift_m * 1000,
            "warning": raw.backend.has_warnings()}


def close_stage(raw, profile: ClosingProfile, prepared: Prepared, cfg: ClosingProtocol, bodies) -> tuple:
    """Hold integrated arm commands fixed; compare only jaw speed, target and contact latch."""
    latch, obs, frames, stop_step = ClosingLatch(profile), prepared.obs, [], None
    for step in range(cfg.closing_steps):
        state = raw.grasp_state
        target = latch.target(state.robot.q_cmd[-1], state.grasped)
        if latch.latched is not None and stop_step is None:
            stop_step = step + 1
        action = np.zeros(6)
        action[-1] = closing_action(state.robot.q_cmd[-1], target, profile, raw.config.task.delta_gripper_max_rad)
        obs, reward, *_ = raw.step(action)
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError("Nonfinite closing probe")
        frames.append(frame(raw, "closing", step, prepared, bodies))
    return obs, target, frames, stop_step


def close_metrics(frames: list[dict], cfg: ClosingProtocol) -> dict:
    """Summarize both force peaks and final bilateral stability; do not equate contact with retention."""
    return {"stable_end": all(f["grasped"] for f in frames[-cfg.stable_before_lift:]),
            "max_cube_shift_xy_mm": max(f["cube_shift_xy_mm"] for f in frames),
            "max_cube_rotation_deg": max(f["cube_rotation_deg"] for f in frames),
            "max_pinch_drift_mm": max(f["pinch_drift_mm"] for f in frames),
            "max_arm_command_drift_rad": max(f["arm_command_drift_rad"] for f in frames),
            "peak_fixed_force_n": max(f["fixed"]["normal_force_n"] for f in frames),
            "peak_moving_force_n": max(f["moving"]["normal_force_n"] for f in frames),
            "final_jaw_deg": frames[-1]["jaw_deg"], "final_command_deg": frames[-1]["jaw_command_deg"],
            "table_steps": sum(f["table_contact"] for f in frames), "warnings": any(f["warning"] for f in frames)}


def lift_stage(raw, policy, obs, target: float, profile, prepared, cfg, bodies) -> list[dict]:
    """Recompute lift from the settled object, retaining the original arm rate and contact-loss abort."""
    goal = raw.grasp_state.obj.p_s + [0, 0, policy._cfg.lift_height_m]
    q = policy._ik(goal, policy._q_cmd(obs), False)
    phase = PhaseTarget(q, target, policy._cfg.lift_rate_rad, profile.rate_rad)
    aborted, frames = False, []
    for step in range(cfg.lift_steps):
        aborted |= not raw.grasp_state.grasped
        action = policy._track(obs, phase)
        if aborted:
            action[:-1] = 0
        obs, reward, *_ = raw.step(action)
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError("Nonfinite lift probe")
        record = frame(raw, "lift", step, prepared, bodies)
        record["success"] = bool(raw._reward_fn.is_success(raw.grasp_state))
        record["aborted"] = aborted
        frames.append(record)
    return frames


def episode(env, policy, seed: int, profile: ClosingProfile, cfg=ClosingProtocol(), trace: bool = False) -> dict:
    """Separate acquisition failure, closing failure, early loss and final task success."""
    prepared = prepare(env, policy, seed, cfg)
    if prepared is None:
        return {"seed": seed, "prepared": False, "success": False, "reason": policy.abort_reason}
    raw = env.unwrapped
    bodies = descendant_body_ids(raw.backend.model, raw.config.sim.robot_root_body)
    obs, target, closed, stop_step = close_stage(raw, profile, prepared, cfg, bodies)
    summary = close_metrics(closed, cfg)
    lifted = lift_stage(raw, policy, obs, target, profile, prepared, cfg, bodies) if summary["stable_end"] else []
    result = {"seed": seed, "prepared": True, "preparation_sha256": prepared.digest,
              "closing": summary, "stop_step": stop_step, "lift_attempted": bool(lifted),
              "success": bool(lifted and lifted[-1]["success"]),
              "early_lift_retained": all(f["grasped"] for f in lifted[:cfg.early_lift_steps]) if lifted else None,
              "loss_step": next((f["stage_step"] for f in lifted if not f["grasped"]), None),
              "final_lift_mm": lifted[-1]["lift_mm"] if lifted else closed[-1]["lift_mm"],
              "table_steps": summary["table_steps"] + sum(f["table_contact"] for f in lifted),
              "warnings": summary["warnings"] or any(f["warning"] for f in lifted)}
    if trace:
        result["trace"] = closed + lifted
    return result


def evaluate(profile: ClosingProfile, seeds) -> list[dict]:
    """Pair the acquisition trajectory and diagnostic time budgets across every profile."""
    rows = []
    for effects in (False, True):
        for mass in PickLiftEvalConfig().masses_kg:
            env, policy = build(mass, effects)
            try:
                episodes = [episode(env, policy, seed, profile) for seed in seeds]
                rows.append({"profile": profile.name, "mass_kg": mass, "arm_effects": effects, "episodes": episodes})
                print(profile.name, effects, mass, "stable", sum(e.get("closing", {}).get("stable_end", False) for e in episodes),
                      "lift", sum(e["success"] for e in episodes), "/", len(episodes), flush=True)
            finally:
                env.close()
    return rows


def metrics(rows: list[dict]) -> dict:
    """Reject table contact, warnings and low overall success, including preparation failures."""
    episodes = [e for row in rows for e in row["episodes"]]
    return {"worst_success": min(float(np.mean([e["success"] for e in r["episodes"]])) for r in rows),
            "table_episodes": sum(e.get("table_steps", 0) > 0 for e in episodes),
            "warnings": sum(e.get("warnings", False) for e in episodes)}


def choose(records: list[dict]) -> dict:
    """Prefer a simple fixed endpoint on ties; freeze before the new validation placements."""
    candidates = records[1:]
    eligible = [r for r in candidates if r["metrics"]["worst_success"] >= .95 and
                not r["metrics"]["table_episodes"] and not r["metrics"]["warnings"] and
                all(e["success"] for row in r["known"] for e in row["episodes"])]
    best = max(eligible or candidates, key=lambda r: (r["metrics"]["worst_success"], -r["metrics"]["table_episodes"]))
    return {"profile": best["profile"], "passed_tuning_gate": bool(eligible), "metrics": best["metrics"]}


def verify_pairing(records: list[dict]) -> None:
    """Assert profiles start from identical prepared physical states for each paired case."""
    seen = {}
    for row in records:
        for e in row["episodes"]:
            if not e["prepared"]:
                continue
            key = (row["arm_effects"], row["mass_kg"], e["seed"])
            digest = e["preparation_sha256"]
            if key in seen and seen[key] != digest:
                raise AssertionError("Unmatched acquisition state across closing profiles")
            seen[key] = digest


def run(output: Path) -> None:
    """Known cases + 76000-76002 tune, freeze, then validate on 77000-77019."""
    output.mkdir(parents=True, exist_ok=False)
    hashes, records = protected_inputs(), []
    write(output / "protocol.json", {"closing": asdict(ClosingProtocol()), "profiles": [asdict(p) for p in PROFILES],
          "known_seeds": [68006, 69005, 75003], "tuning_seeds": [76000, 76002], "heldout_seeds": [77000, 77019],
          "protected_sha256": hashes, "gate": ">=95% final success every mass/mode; zero table contact and warnings; known cases lift"})
    for profile in PROFILES:
        known, rows = evaluate(profile, (68006, 69005, 75003)), evaluate(profile, range(76000, 76003))
        records.append({"profile": asdict(profile), "known": known, "rows": rows, "metrics": metrics(rows)})
        write(output / "tuning.json", records)
    verify_pairing([r for record in records for r in record["known"] + record["rows"]])
    selection = choose(records)
    write(output / "frozen_selection.json", selection)
    heldout = []
    for profile in (PROFILES[0], ClosingProfile(**selection["profile"])):
        heldout += evaluate(profile, range(77000, 77020))
        write(output / "heldout.json", heldout)
    verify_pairing(heldout)
    selected = [r for r in heldout if r["profile"] == selection["profile"]["name"]]
    result = metrics(selected)
    passed = result["worst_success"] >= .95 and not result["table_episodes"] and not result["warnings"]
    write(output / "assessment.json", {"metrics": result, "passed_heldout_gate": passed,
          "overall_validated": bool(passed and selection["passed_tuning_gate"]),
          "promote_defaults": False, "train": False, "pairing_verified": True})
    assert hashes == protected_inputs(), "Original artifacts changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


def evaluate_end_to_end(profile: ClosingProfile | None, label: str, seeds) -> list[dict]:
    """Return to the original 200-step task; compare the endpoint without extra hold time."""
    from scripts.sim2real.analyze.raised_approach_check import BASELINE, build as old_build, episode as acquisition_episode

    rows = []
    for effects in (False, True):
        for mass in PickLiftEvalConfig().masses_kg:
            env, policy = old_build(BASELINE, mass, effects) if profile is None else build_end_to_end(profile, mass, effects)
            try:
                episodes = [asdict(acquisition_episode(env, policy, seed)) for seed in seeds]
                rows.append({"variant": label, "mass_kg": mass, "arm_effects": effects, "episodes": episodes})
                print(label, effects, mass, "lift", sum(e["success"] for e in episodes), "/", len(episodes),
                      "table", sum(e["table_steps"] > 0 for e in episodes), flush=True)
            finally:
                env.close()
    return rows


def end_to_end(output: Path, selection_path: Path) -> None:
    """Freeze selected profile before fresh 78000-78019 original-horizon validation."""
    from scripts.sim2real.analyze.raised_approach_check import metrics as acquisition_metrics, passes

    output.mkdir(parents=True, exist_ok=False)
    profile = ClosingProfile(**json.loads(selection_path.read_text())["profile"])
    hashes = protected_inputs()
    write(output / "protocol.json", {"profile": asdict(profile), "known_seeds": [68006, 69005, 75003],
          "heldout_seeds": [78000, 78019], "episode_steps": 200, "protected_sha256": hashes,
          "gate": ">=95% every mass/mode, known regressions all lift, zero table/approach contact and warnings, <=5% >1mm preclose shifts"})
    records = []
    for label, candidate in (("legacy", None), ("cartesian_minus10", PROFILES[0]), ("cartesian_selected", profile)):
        known = evaluate_end_to_end(candidate, label, (68006, 69005, 75003))
        heldout = evaluate_end_to_end(candidate, label, range(78000, 78020))
        records.append({"variant": label, "known": known, "heldout": heldout, "metrics": acquisition_metrics(heldout)})
        write(output / "comparisons.json", records)
    known_pass = all(e["success"] for row in records[-1]["known"] for e in row["episodes"])
    write(output / "assessment.json", {"known_regressions_pass": known_pass,
          "heldout_metrics": records[-1]["metrics"], "passed": bool(known_pass and passes(records[-1]["metrics"])),
          "promote_defaults": False, "train": False})
    assert hashes == protected_inputs(), "Original artifacts changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


def replay(output: Path, selection_path: Path, mass: float, seed: int, viewer: bool) -> None:
    """Replay the selected fixed endpoint in the original 200-step task."""
    from scripts.sim2real.analyze.replay_pick_lift import rollout, transition_events

    output.mkdir(parents=True, exist_ok=False)
    profile = ClosingProfile(**json.loads(selection_path.read_text())["profile"])
    env, policy = build_end_to_end(profile, mass, True, "human" if viewer else None)
    try:
        trace = rollout(env, policy, seed, output, frames=True, viewer=viewer)
        write(output / "trace.json", {"profile": asdict(profile), "mass_kg": mass, "seed": seed,
              "episode_steps": 200, "events": transition_events(trace), "trace": trace})
        print("Replay final success", trace[-1]["success"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--end-to-end", type=Path, help="Frozen diagnostic selection to validate on fresh 200-step cases")
    mode.add_argument("--replay", type=Path, help="Frozen selection to replay at the original task horizon")
    parser.add_argument("--mass-kg", type=float, default=.005)
    parser.add_argument("--placement-seed", type=int, default=68006)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.end_to_end:
        end_to_end(args.out, args.end_to_end)
    elif args.replay:
        replay(args.out, args.replay, args.mass_kg, args.placement_seed, args.viewer)
    else:
        run(args.out)

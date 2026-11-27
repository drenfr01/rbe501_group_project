"""Frozen baseline, acquisition contact diagnosis and paired holding-force probes."""

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from rbe_501_so101.config import PickLiftEvalConfig
from rbe_501_so101.envs.grasp_model import FIXED_PAD_GEOM, MOVING_PAD_GEOM
from rbe_501_so101.envs.grasp_servo_backend import GraspServoBackend
from rbe_501_so101.rl.limited_lift_training import LimitedLiftRepository, load_policy, make_env
from scripts.sim2real.analyze.calibrated_lift_pilot import ZeroResidual, write
from scripts.sim2real.analyze.limited_lift_pilot import protected_inputs
from scripts.sim2real.analyze.replay_pick_lift import contact_records, phase_at


@dataclass(frozen=True)
class Pulse:
    """World force on the cube COM, in N, for three 33 ms control periods."""

    name: str
    force_n: tuple[float, float, float]
    first_step: int = 150
    duration_steps: int = 3


@dataclass(frozen=True)
class HoldingProtocol:
    """Independent placements, stable precondition, recovery metric and CI seed."""

    first_seed: int = 69_000
    placements: int = 10
    stable_steps: int = 10
    final_window: int = 30
    bootstrap_seed: int = 0
    bootstrap_samples: int = 10_000


PULSES = (Pulse("none", (0., 0., 0.)), *(Pulse(f"{force:g}N_{axis}", vector)
          for force in (.1, .2) for axis, vector in (
              ("+x", (force, 0., 0.)), ("-x", (-force, 0., 0.)),
              ("+y", (0., force, 0.)), ("-y", (0., -force, 0.)))))


def force_at(step: int, pulse: Pulse) -> np.ndarray:
    """Use zero-based steps; exactly duration_steps periods receive the pulse."""
    active = pulse.first_step <= step < pulse.first_step + pulse.duration_steps
    return np.asarray(pulse.force_n) if active else np.zeros(3)


@contextmanager
def temporary_force(data, body: int, force: np.ndarray):
    """Add a world-space force and restore the original wrench even after failure."""
    previous = data.xfrc_applied[body].copy()
    data.xfrc_applied[body, :3] += force
    try:
        yield
    finally:
        data.xfrc_applied[body] = previous


def displacement_rms(positions: np.ndarray, reference: np.ndarray) -> float:
    """RMS 3-D displacement from one fixed pre-pulse pose, in meters."""
    return float(np.sqrt(np.mean(np.sum((positions - reference) ** 2, axis=1))))


def physical_state(raw):
    """Measure true simulated poses, separate from delayed/quantized observations."""
    backend = raw.backend.inner if isinstance(raw.backend, GraspServoBackend) else raw.backend
    return backend.grasp_state()


def fixed_env(preset, mass: float, effects: bool | None = None):
    """Keep the baseline fixed except the explicit diagnostic mass/effects condition."""
    base = replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass))
    if effects is not None:
        base = replace(base, arm_effects=effects)
    return make_env(replace(preset, base=base))


def pad_record(raw, name: str, contacts: list[dict]) -> dict:
    """Signed pad/cube gap and actual contact forces/locations in the cube frame."""
    model, data = raw.backend.model, raw.backend.data
    cube, pad = model.geom(raw.config.object.geom).id, model.geom(name).id
    gap = mujoco.mj_geomDistance(model, data, pad, cube, .1, None)
    body = model.body(raw.config.object.body).id
    rotation = data.xmat[body].reshape(3, 3)
    selected = [c for c in contacts if name in c["geoms"] and "cube" in c["bodies"]]
    local = [(rotation.T @ (np.asarray(c["position_m"]) - data.xpos[body]) * 1000).tolist() for c in selected]
    return {"gap_mm": float(gap * 1000), "contacts": len(selected),
            "normal_force_n": sum(c["normal_force_n"] for c in selected), "cube_local_contacts_mm": local}


def acquisition_frame(env, step: int) -> dict:
    """Record cube motion, measured pinch error, jaw angle and both pad contacts."""
    raw = env.unwrapped
    state = physical_state(raw)
    contacts = contact_records(raw)
    q = raw.backend.data.qpos[raw.backend.ids.qpos_adr]
    return {"step": step + 1, "phase": phase_at(step), "cube_m": state.obj.p_s.tolist(),
            "pinch_minus_cube_mm": ((state.p_pinch - state.obj.p_s) * 1000).tolist(),
            "jaw_deg": float(np.rad2deg(q[-1])), "grasped": state.grasped,
            "fixed": pad_record(raw, FIXED_PAD_GEOM, contacts),
            "moving": pad_record(raw, MOVING_PAD_GEOM, contacts)}


def acquisition_episode(env, seed: int) -> dict:
    """Trace only acquisition; retain the final unmodified task outcome."""
    obs, _ = env.reset(seed=seed)
    frames = []
    for step in range(env.unwrapped.config.task.max_episode_steps):
        obs, _, _, _, info = env.step(np.zeros(6))
        if step < 100:
            frames.append(acquisition_frame(env, step))
    return {"seed": seed, "frames": frames, "final_success": bool(info["is_success"]),
            "table_steps": info["robot_table_contact_steps"], "warning": info["mujoco_warning"]}


def diagnose(preset, output: Path) -> None:
    """Compare failed placement to a known working control, across mass and effects."""
    rows = []
    for effects in (False, True):
        for mass in PickLiftEvalConfig().masses_kg:
            env = fixed_env(preset, mass, effects)
            try:
                rows += [{"arm_effects": effects, "mass_kg": mass, **acquisition_episode(env, s)}
                         for s in (68000, 68006)]
            finally:
                env.close()
    write(output / "acquisition.json", rows)
    print("Acquisition diagnostics complete:", sum(r["final_success"] for r in rows), "/", len(rows), flush=True)


def holding_metrics(cubes, pinches, reference, protocol: HoldingProtocol) -> dict:
    """Include drops in position metrics; do not censor disturbed failures."""
    cube, pinch = np.asarray(cubes), np.asarray(pinches)
    pre_cube, pre_pinch = reference
    slip = (cube - pinch) - (pre_cube - pre_pinch)
    return {"cube_rms_mm": displacement_rms(cube[-protocol.final_window:], pre_cube) * 1000,
            "pinch_rms_mm": displacement_rms(pinch[-protocol.final_window:], pre_pinch) * 1000,
            "peak_cube_mm": float(np.linalg.norm(cube - pre_cube, axis=1).max() * 1000),
            "peak_slip_mm": float(np.linalg.norm(slip, axis=1).max() * 1000)}


def holding_episode(env, policy, seed: int, pulse: Pulse, protocol: HoldingProtocol) -> dict:
    """Perturb only a stable lift; record ineligible episodes in the overall denominator."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    raw = env.unwrapped
    body = raw.backend.model.body(raw.config.object.body).id
    history, cubes, pinches = [], [], []
    eligible = lost = False
    reference = None
    for step in range(raw.config.task.max_episode_steps):
        if step == pulse.first_step:
            eligible = len(history) >= protocol.stable_steps and all(history[-protocol.stable_steps:])
            state = physical_state(raw)
            reference = (state.obj.p_s.copy(), state.p_pinch.copy())
        force = force_at(step, pulse) if eligible else np.zeros(3)
        with temporary_force(raw.backend.data, body, force):
            obs, reward, _, _, info = env.step(policy(obs))
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError("Nonfinite holding probe state")
        history.append(bool(info["is_success"]))
        if step >= pulse.first_step:
            state = physical_state(raw)
            cubes.append(state.obj.p_s.copy())
            pinches.append(state.p_pinch.copy())
            lost |= eligible and not bool(info["is_success"])
    metrics = holding_metrics(cubes, pinches, reference, protocol) if eligible else None
    return {"seed": seed, "eligible": eligible, "success": bool(info["is_success"]),
            "lost_success": lost, "table_steps": info["robot_table_contact_steps"],
            "warning": info["mujoco_warning"], "metrics": metrics, "interventions": env.counters.copy()}


def frozen_inputs(repo: LimitedLiftRepository) -> dict:
    """Freeze script source, physics and both pilot policies before any new probe."""
    result = protected_inputs()
    paths = [repo.path, repo.root / "seed_0/model.zip", repo.root / "seed_0/vecnormalize.pkl",
             Path("src/rbe_501_so101/control/aligned_pick_lift.py"),
             Path("src/rbe_501_so101/rl/lift_residual_limits.py")]
    result.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return result


def holding_sweep(repo: LimitedLiftRepository, output: Path, protocol: HoldingProtocol) -> list[dict]:
    """Pair controllers, force directions, magnitudes, masses and placement seeds."""
    preset, learned = repo.load(), load_policy(repo)
    rows = []
    try:
        for pulse in PULSES:
            for name, policy in (("scripted", ZeroResidual()), ("limited_sac", learned)):
                for mass in PickLiftEvalConfig().masses_kg:
                    env = fixed_env(preset, mass)
                    try:
                        rows += [{"controller": name, "mass_kg": mass, "pulse": pulse.name,
                                  **holding_episode(env, policy, protocol.first_seed + n, pulse, protocol)}
                                 for n in range(protocol.placements)]
                    finally:
                        env.close()
                write(output / "holding_episodes.json", rows)
                latest = [r for r in rows if r["pulse"] == pulse.name and r["controller"] == name]
                print(pulse.name, name, "success", sum(r["success"] for r in latest),
                      "eligible", sum(r["eligible"] for r in latest), flush=True)
    finally:
        learned.close()
    return rows


def paired_results(rows: list[dict]) -> list[dict]:
    """Subtract each controller's no-force drift, using common eligible pairs only."""
    index = {(r["controller"], r["mass_kg"], r["seed"], r["pulse"]): r for r in rows}
    pairs = []
    for r in rows:
        if r["controller"] != "scripted" or r["pulse"] == "none":
            continue
        mass, seed, pulse = r["mass_kg"], r["seed"], r["pulse"]
        group = [index[(c, mass, seed, p)] for c, p in (("scripted", pulse), ("limited_sac", pulse),
                                                       ("scripted", "none"), ("limited_sac", "none"))]
        if not all(g["eligible"] for g in group):
            continue
        a, b, az, bz = (g["metrics"]["cube_rms_mm"] for g in group)
        pairs.append({"mass_kg": mass, "seed": seed, "pulse": pulse,
                      "adjusted_cube_rms_difference_mm": (b - bz) - (a - az),
                      "raw_cube_rms_difference_mm": b - a,
                      "script_success": group[0]["success"], "rl_success": group[1]["success"]})
    return pairs


def primary_assessment(pairs: list[dict], protocol: HoldingProtocol) -> dict:
    """Bootstrap placement-cluster means; negative adjusted RMS difference favors RL."""
    seeds = sorted({p["seed"] for p in pairs})
    means = np.array([np.mean([p["adjusted_cube_rms_difference_mm"] for p in pairs if p["seed"] == s]) for s in seeds])
    if not len(means):
        return {"benefit_established": False, "reason": "No common eligible placement pairs"}
    rng = np.random.default_rng(protocol.bootstrap_seed)
    samples = rng.choice(means, (protocol.bootstrap_samples, len(means)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [.025, .975])
    return {"placement_clusters": len(seeds), "paired_cases": len(pairs),
            "mean_adjusted_difference_mm": float(means.mean()), "ci95_mm": [float(lo), float(hi)],
            "precision_benefit_observed": bool(hi < 0),
            "note": "One trained seed; repeated directions/masses are clustered by placement, not independent cases."}


def run(repo: LimitedLiftRepository, output: Path) -> None:
    """Freeze a new evaluation protocol; never train, tune a controller or edit a model."""
    output.mkdir(parents=True, exist_ok=False)
    protocol, hashes = HoldingProtocol(), frozen_inputs(repo)
    write(output / "protocol.json", {"holding": asdict(protocol), "pulses": [asdict(p) for p in PULSES],
          "baseline_sha256": hashes, "preset": asdict(repo.load()),
          "primary": "Force-minus-no-force final-window cube RMS; paired RL-minus-script difference, bootstrap by placement.",
          "benefit_rule": "95% CI entirely below zero, no worse eligible retention and no additional table-contact episodes."})
    diagnose(repo.load(), output)
    rows = holding_sweep(repo, output, protocol)
    pairs = paired_results(rows)
    assessment = primary_assessment(pairs, protocol)
    retention = all(p["rl_success"] or not p["script_success"] for p in pairs)
    table_ok = sum(r["table_steps"] > 0 for r in rows if r["controller"] == "limited_sac") <= sum(r["table_steps"] > 0 for r in rows if r["controller"] == "scripted")
    assessment.update(retention_not_worse=retention, table_contact_not_worse=table_ok,
                      benefit_established=bool(assessment.get("precision_benefit_observed") and retention and table_ok))
    write(output / "paired_results.json", pairs)
    write(output / "assessment.json", assessment)
    assert hashes == frozen_inputs(repo), "A frozen baseline input changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(LimitedLiftRepository(args.model_dir), args.out)

"""Frozen full-task policy versus scripted baseline under paired small carry pushes."""

import argparse
import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.envs.pick_place_env import SO101PickPlaceEnv
from rbe_501_so101.rl.pick_place_training import PickPlaceRepository, load_policy
from scripts.sim2real.analyze.acquisition_holding_checks import temporary_force
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.pick_place_pilot import ZeroPolicy


@dataclass(frozen=True)
class CarryProtocol:
    """Predeclared moderate forces during the first carrying motion, with no model variation."""

    first_seed: int = 94000
    placements: int = 10
    masses_kg: tuple[float, ...] = (.005, .01, .015, .02)
    yaw_deg: float = 0.
    forces_n: tuple[float, ...] = (.1, .2)
    pulse_step: int = 200
    pulse_periods: int = 3
    stable_periods: int = 5
    minimum_lift_m: float = .04
    post_periods: int = 30
    recovery_tolerance_m: float = .001
    practical_placement_gain_mm: float = .5
    bootstrap_seed: int = 0
    bootstrap_samples: int = 10000


@dataclass(frozen=True)
class CarryPulse:
    """World-space force at the cube COM, without added torque."""

    name: str
    force_n: tuple[float, float, float]


@dataclass(frozen=True)
class CarryOutcome:
    """Keep all acquisitions and final failures; path windows are separate secondary evidence."""

    controller: str
    seed: int
    mass_kg: float
    pulse: str
    eligible: bool
    success: bool
    final_error_mm: float
    table_steps: int
    warning: bool
    transport_grasp_loss: bool
    force_periods: int
    force_duration_s: float
    pulse_phases: tuple[str, ...]
    failure_reason: str
    episode_steps: int
    residual_steps: int
    cube_window: tuple[tuple[float, ...], ...]
    pinch_window: tuple[tuple[float, ...], ...]
    grasp_window: tuple[bool, ...]


def pulses(cfg: CarryProtocol) -> tuple[CarryPulse, ...]:
    """Include a no-force control and the full fixed magnitude/direction matrix."""
    return (CarryPulse("none", (0., 0., 0.)), *(CarryPulse(f"{f:g}N_{axis}", vector)
        for f in cfg.forces_n for axis, vector in (("+x", (f, 0., 0.)), ("-x", (-f, 0., 0.)),
                                                  ("+y", (0., f, 0.)), ("-y", (0., -f, 0.)))))


def pulse_force(step: int, pulse: CarryPulse, cfg: CarryProtocol) -> np.ndarray:
    """Use exactly the same absolute control-period schedule in both pipelines."""
    active = cfg.pulse_step <= step < cfg.pulse_step + cfg.pulse_periods
    return np.asarray(pulse.force_n) if active else np.zeros(3)


def eligible(phase: str, stable_history: list[bool], cfg: CarryProtocol) -> bool:
    """Require five established, lifted and unsupported grasp samples before carrying push."""
    return (phase == "carry" and len(stable_history) >= cfg.stable_periods
            and all(stable_history[-cfg.stable_periods:]))


def recovery_periods(deviations: list[float], grasped: list[bool], cfg: CarryProtocol) -> int | None:
    """Confirm five consecutive held samples within 1mm of the own-controller nominal grip path."""
    count = 0
    for step, (distance, held) in enumerate(zip(deviations, grasped), 1):
        count = count + 1 if held and distance <= cfg.recovery_tolerance_m else 0
        if count >= cfg.stable_periods:
            return step
    return None


def episode(env, policy, name: str, seed: int, pulse: CarryPulse, cfg: CarryProtocol) -> CarryOutcome:
    """Apply only eligible pushes and retain actual phases and complete-task outcome."""
    obs, _ = env.reset(seed=seed, options={"yaw_offset_deg": cfg.yaw_deg})
    policy.reset()
    history, cubes, pinches, held, phases = [], [], [], [], []
    armed = lost = False
    applied = 0
    duration = 0.
    seq = env.sequence
    for step in range(env.config.max_episode_steps):
        if step == cfg.pulse_step:
            armed = eligible(str(env.stepper.phase), history, cfg)
        active = armed and cfg.pulse_step <= step < cfg.pulse_step + cfg.pulse_periods
        force = pulse_force(step, pulse, cfg) if armed else np.zeros(3)
        previous_time = seq.backend.data.time
        with temporary_force(seq.backend.data, seq.cube, force):
            obs, reward, terminated, truncated, info = env.step(policy(obs))
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError("Nonfinite disturbed full-task rollout")
        state = seq.physical.grasp_state()
        phase = seq.frames[-1].phase
        history.append(bool(state.grasped and state.obj.lift_m >= cfg.minimum_lift_m and not seq.contacts()[0]))
        lost |= phase in ("carry", "lower") and not state.grasped
        if active:
            elapsed = seq.backend.data.time - previous_time
            duration += elapsed
            if elapsed > 0:
                applied += 1
                phases.append(phase)
        if cfg.pulse_step <= step < cfg.pulse_step + cfg.pulse_periods + cfg.post_periods:
            cubes.append(tuple(state.obj.p_s))
            pinches.append(tuple(state.p_pinch))
            held.append(bool(state.grasped))
        if terminated or truncated:
            break
    return CarryOutcome(name, seed, info["mass_kg"], pulse.name, armed, bool(info["is_success"]),
        info["placement_error_m"] * 1000, info["robot_table_contact_steps"], info["mujoco_warning"],
        bool(lost or info["failure_reason"] == "grasp_lost_during_transport"), applied, duration, tuple(phases),
        info["failure_reason"], info["episode_steps"], info["residual_steps"], tuple(cubes), tuple(pinches), tuple(held))


def sweep(repo, output: Path, cfg: CarryProtocol) -> list[CarryOutcome]:
    """Pair all placements, masses and pulses; freeze weights and observation statistics."""
    base, learned = repo.load(), load_policy(repo)
    rows = []
    try:
        for pulse in pulses(cfg):
            for name, policy in (("scripted", ZeroPolicy()), ("sac", learned)):
                for mass in cfg.masses_kg:
                    preset = replace(base.preset, base=replace(base.preset.base,
                        mass=replace(base.preset.base.mass, mode="fixed", fixed_kg=mass)))
                    env = SO101PickPlaceEnv(replace(base, preset=preset))
                    try:
                        rows.extend(episode(env, policy, name, seed, pulse, cfg)
                                    for seed in range(cfg.first_seed, cfg.first_seed + cfg.placements))
                    finally:
                        env.close()
                write(output / "episodes.json", [asdict(r) for r in rows])
                latest = [r for r in rows if r.pulse == pulse.name and r.controller == name]
                print(pulse.name, name, "complete", sum(r.success for r in latest), "/", len(latest),
                      "eligible", sum(r.eligible for r in latest), flush=True)
    finally:
        learned.close()
    return rows


def cluster_interval(values: list[tuple[int, float]], cfg: CarryProtocol) -> dict:
    """Bootstrap placement-cluster means, not correlated mass/force repetitions."""
    seeds = sorted({s for s, _ in values})
    if not seeds:
        return {"clusters": 0, "mean": None, "ci95": [None, None]}
    means = np.array([np.mean([v for s, v in values if s == seed]) for seed in seeds])
    rng = np.random.default_rng(cfg.bootstrap_seed)
    draws = rng.choice(means, (cfg.bootstrap_samples, len(means)), replace=True).mean(axis=1)
    return {"clusters": len(seeds), "mean": float(means.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist()}


def path_metrics(forced: CarryOutcome, zero: CarryOutcome, cfg: CarryProtocol) -> dict | None:
    """Compare matched physical paths; do not pad unobserved motion after early termination."""
    expected = cfg.pulse_periods + cfg.post_periods
    if len(forced.cube_window) != expected or len(zero.cube_window) != expected:
        return None
    cube = np.asarray(forced.cube_window) - np.asarray(zero.cube_window)
    relative = ((np.asarray(forced.cube_window) - np.asarray(forced.pinch_window))
                - (np.asarray(zero.cube_window) - np.asarray(zero.pinch_window)))
    distances = np.linalg.norm(relative[cfg.pulse_periods:], axis=1)
    held = [a and b for a, b in zip(forced.grasp_window[cfg.pulse_periods:], zero.grasp_window[cfg.pulse_periods:])]
    return {"path_rms_mm": float(np.sqrt(np.mean(np.sum(cube**2, axis=1))) * 1000),
            "peak_path_deviation_mm": float(np.linalg.norm(cube, axis=1).max() * 1000),
            "peak_grip_relative_deviation_mm": float(np.linalg.norm(relative, axis=1).max() * 1000),
            "confirmed_recovery_periods": recovery_periods(distances.tolist(), held, cfg)}


def comparisons(rows: list[CarryOutcome], cfg: CarryProtocol) -> list[dict]:
    """Retain complete-task failures in paired outcomes and subtract each no-force final error."""
    index = {(r.controller, r.seed, r.mass_kg, r.pulse): r for r in rows}
    pairs = []
    for a in rows:
        if a.controller != "scripted" or a.pulse == "none":
            continue
        b = index[("sac", a.seed, a.mass_kg, a.pulse)]
        az, bz = (index[(c, a.seed, a.mass_kg, "none")] for c in ("scripted", "sac"))
        common = all(r.eligible for r in (a, b, az, bz))
        pairs.append({"seed": a.seed, "mass_kg": a.mass_kg, "pulse": a.pulse, "common_eligible": common,
            "script_success": a.success, "rl_success": b.success,
            "adjusted_final_error_difference_mm": (b.final_error_mm - bz.final_error_mm) - (a.final_error_mm - az.final_error_mm),
            "script_path": path_metrics(a, az, cfg), "rl_path": path_metrics(b, bz, cfg)})
    return pairs


def assess(rows: list[CarryOutcome], pairs: list[dict], cfg: CarryProtocol) -> dict:
    """Require meaningful paired benefit without extra failures or sampled safety regressions."""
    eligible_pairs = [p for p in pairs if p["common_eligible"]]
    extra = [p for p in pairs if p["script_success"] and not p["rl_success"]]
    retention = cluster_interval([(p["seed"], float(p["rl_success"]) - float(p["script_success"])) for p in eligible_pairs], cfg)
    error = cluster_interval([(p["seed"], p["adjusted_final_error_difference_mm"]) for p in eligible_pairs], cfg)
    expected = cfg.placements * len(cfg.masses_kg) * (len(pulses(cfg)) - 1)
    identities = {(r.controller, r.seed, r.mass_kg, r.pulse) for r in rows}
    complete = len(pairs) == expected and len(eligible_pairs) == expected and len(identities) == cfg.placements * len(cfg.masses_kg) * len(pulses(cfg)) * 2
    phase_valid = all(all(p == "carry" for p in r.pulse_phases) for r in rows)
    controls = [r for r in rows if r.pulse == "none"]
    controls_pass = bool(controls) and all(r.success for r in controls)
    index = {(r.controller, r.seed, r.mass_kg): r for r in controls}
    extra_controls = sum(r.success and not index[("sac", r.seed, r.mass_kg)].success for r in controls if r.controller == "scripted")
    safe = not any(r.table_steps or r.warning for r in rows if r.controller == "sac")
    gain = retention["clusters"] > 1 and retention["ci95"][0] > 0
    precision = error["clusters"] > 1 and error["ci95"][1] < -cfg.practical_placement_gain_mm
    return {"episodes": len(rows), "common_eligible_disturbed_pairs": len(eligible_pairs),
            "complete_protocol": complete, "pulse_phases_valid": phase_valid, "controls_pass": controls_pass,
            "additional_rl_failures": len(extra), "additional_rl_control_failures": extra_controls, "safe": safe,
            "retention_rl_minus_script": retention, "adjusted_final_error_difference_mm": error,
            "useful_benefit_established": bool(complete and phase_valid and controls_pass and safe and not extra and not extra_controls and (gain or precision)),
            "start_long_training": False}


def fingerprints(repo: PickPlaceRepository) -> dict[str, str]:
    """Protect exact full-task weights/statistics/config and runtime/model/calibration snapshot."""
    repo.load()
    from rbe_501_so101.rl.pick_place_training import input_manifest
    hashes = input_manifest(repo.load())
    for path in (repo.path, repo.root / "seed_0/model.zip", repo.root / "seed_0/vecnormalize.pkl"):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def run(repo: PickPlaceRepository, output: Path) -> None:
    """Save the complete test protocol first; never train, tune or replace saved artifacts."""
    output.mkdir(parents=True, exist_ok=False)
    cfg, hashes = CarryProtocol(), fingerprints(repo)
    write(output / "protocol.json", {"protocol": asdict(cfg), "pulses": [asdict(p) for p in pulses(cfg)],
        "input_sha256": hashes, "primary": "Paired complete-placement retention on common eligible carry pushes.",
        "benefit_rule": "Retention CI >0 or adjusted placement-error CI better than -0.5mm; no additional paired failures or policy table contacts/warnings.",
        "recovery": "Five held samples within 1mm of own unforced grip-relative path, after force removal; minimum confirmation 165ms.",
        "limits": "Nominal cube yaw and nominal fixed arm effects; one frozen training seed; no model randomization.",
        "early_failure": "Retain pre-eligible trials that fail during a partial pulse in success comparisons; never discard a force-induced failure for shortened exposure."})
    rows = sweep(repo, output, cfg)
    pairs = comparisons(rows, cfg)
    write(output / "paired_results.json", pairs)
    write(output / "assessment.json", assess(rows, pairs, cfg))
    assert hashes == fingerprints(repo), "A frozen runtime/model/policy input changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/pick_place_v1_pilot"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(PickPlaceRepository(args.model_dir), args.out)

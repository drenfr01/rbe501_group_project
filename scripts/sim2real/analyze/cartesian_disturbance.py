"""Frozen v3 SAC versus scripted lift robustness; no training or controller changes."""

import argparse
import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository, load_policy, make_env
from scripts.sim2real.analyze.acquisition_holding_checks import (
    HoldingProtocol, Pulse, force_at, holding_metrics, physical_state, temporary_force,
)
from scripts.sim2real.analyze.calibrated_lift_pilot import ZeroResidual, write
from scripts.sim2real.analyze.cartesian_zero_pilot import protected_inputs
from scripts.sim2real.analyze.raised_approach_check import touches_cube
from rbe_501_so101.envs.model_indices import descendant_body_ids


@dataclass(frozen=True)
class RobustnessProtocol(HoldingProtocol):
    """Predeclare new placements and practical benefit thresholds before evaluation."""

    first_seed: int = 81_000
    forces_n: tuple[float, ...] = (.1, .2, .4, .8)
    masses_kg: tuple[float, ...] = (.005, .01, .015, .02)
    practical_cube_improvement_mm: float = .05
    permitted_pinch_increase_mm: float = .05


@dataclass(frozen=True)
class DisturbanceOutcome:
    """Retain acquisition failures, transient losses, recovery and actual physical motion."""

    controller: str
    mass_kg: float
    seed: int
    pulse: str
    eligible: bool
    success: bool
    lost_success: bool
    lost_grasp: bool
    approach_contact: bool
    table_steps: int
    warning: bool
    recovery_steps: int | None
    cube_rms_mm: float | None
    pinch_rms_mm: float | None
    peak_cube_mm: float | None
    peak_slip_mm: float | None


def pulses(protocol: RobustnessProtocol) -> tuple[Pulse, ...]:
    """Apply a fixed force ladder in four world directions, plus a no-force control."""
    return (Pulse("none", (0., 0., 0.)), *(Pulse(f"{f:g}N_{axis}", vector)
            for f in protocol.forces_n for axis, vector in (
                ("+x", (f, 0., 0.)), ("-x", (-f, 0., 0.)),
                ("+y", (0., f, 0.)), ("-y", (0., -f, 0.)))))


def recovery_steps(successes: list[bool], stable_steps: int) -> int | None:
    """Periods after force removal until a full stable-success window has elapsed."""
    consecutive = 0
    for step, success in enumerate(successes, 1):
        consecutive = consecutive + 1 if success else 0
        if consecutive >= stable_steps:
            return step
    return None


def finish_outcome(env, name, mass, seed, pulse, protocol, records, eligible, reference):
    """Summarize without dropping failed disturbed holds from the motion metrics."""
    history, grasped, cubes, pinches, approach = records
    first, end = pulse.first_step, pulse.first_step + pulse.duration_steps
    metrics = holding_metrics(cubes, pinches, reference, protocol) if eligible else {}
    info = env.last_info
    return DisturbanceOutcome(name, mass, seed, pulse.name, eligible, history[-1],
        eligible and not all(history[first:]), eligible and not all(grasped[first:]),
        approach, int(info["robot_table_contact_steps"]), bool(info["mujoco_warning"]),
        recovery_steps(history[end:], protocol.stable_steps) if eligible else None,
        *(metrics.get(k) for k in ("cube_rms_mm", "pinch_rms_mm", "peak_cube_mm", "peak_slip_mm")))


def episode(env, policy, name, mass, seed, pulse, protocol) -> DisturbanceOutcome:
    """Apply a COM force only after ten consecutive true successful hold samples."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    raw = env.unwrapped
    body = raw.backend.model.body(raw.config.object.body).id
    bodies = descendant_body_ids(raw.backend.model, raw.config.sim.robot_root_body)
    history, grasped, cubes, pinches = [], [], [], []
    eligible = approach = False
    reference = None
    for step in range(raw.config.task.max_episode_steps):
        if step == pulse.first_step:
            eligible = len(history) >= protocol.stable_steps and all(history[-protocol.stable_steps:])
            state = physical_state(raw)
            reference = state.obj.p_s.copy(), state.p_pinch.copy()
        with temporary_force(raw.backend.data, body, force_at(step, pulse) if eligible else np.zeros(3)):
            obs, reward, _, _, info = env.step(policy(obs))
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError("Nonfinite disturbed rollout")
        history.append(bool(info["is_success"]))
        grasped.append(bool(info["grasped"]))
        approach |= step < 40 and touches_cube(raw, bodies)
        if step >= pulse.first_step:
            state = physical_state(raw)
            cubes.append(state.obj.p_s.copy())
            pinches.append(state.p_pinch.copy())
    env.last_info = info
    return finish_outcome(env, name, mass, seed, pulse, protocol,
                          (history, grasped, cubes, pinches, approach), eligible, reference)


def sweep(repo: CartesianZeroRepository, output: Path, protocol: RobustnessProtocol) -> list[DisturbanceOutcome]:
    """Run the exact same preset and paired conditions for both frozen controllers."""
    preset, learned = repo.load(), load_policy(repo)
    rows = []
    try:
        for pulse in pulses(protocol):
            for name, policy in (("scripted", ZeroResidual()), ("sac", learned)):
                for mass in protocol.masses_kg:
                    base = replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass))
                    env = make_env(replace(preset, base=base))
                    try:
                        rows.extend(episode(env, policy, name, mass, protocol.first_seed + n, pulse, protocol)
                                    for n in range(protocol.placements))
                    finally:
                        env.close()
                write(output / "episodes.json", [asdict(r) for r in rows])
                recent = [r for r in rows if r.controller == name and r.pulse == pulse.name]
                print(pulse.name, name, "eligible", sum(r.eligible for r in recent),
                      "success", sum(r.success for r in recent), flush=True)
    finally:
        learned.close()
    return rows


def matched(rows: list[DisturbanceOutcome]) -> list[tuple[DisturbanceOutcome, DisturbanceOutcome]]:
    """Keep condition keys explicit; preserve acquisition failures in overall summaries."""
    index = {(r.controller, r.mass_kg, r.seed, r.pulse): r for r in rows}
    return [(r, index[("sac", r.mass_kg, r.seed, r.pulse)]) for r in rows
            if r.controller == "scripted" and ("sac", r.mass_kg, r.seed, r.pulse) in index]


def interval(values: list[tuple[int, float]], protocol: RobustnessProtocol) -> dict:
    """Bootstrap placement-cluster means, not correlated mass/direction repetitions."""
    seeds = sorted({s for s, _ in values})
    if not seeds:
        return {"clusters": 0, "mean": None, "ci95": [None, None]}
    means = np.array([np.mean([v for s, v in values if s == seed]) for seed in seeds])
    rng = np.random.default_rng(protocol.bootstrap_seed)
    samples = rng.choice(means, (protocol.bootstrap_samples, len(means)), replace=True).mean(axis=1)
    return {"clusters": len(seeds), "mean": float(means.mean()),
            "ci95": np.quantile(samples, [.025, .975]).tolist()}


def precision_pairs(rows, pairs):
    """Subtract each controller's no-force drift; include failures with eligible acquisition."""
    index = {(r.controller, r.mass_kg, r.seed, r.pulse): r for r in rows}
    result = []
    for a, b in pairs:
        az = index.get(("scripted", a.mass_kg, a.seed, "none"))
        bz = index.get(("sac", a.mass_kg, a.seed, "none"))
        if az is None or bz is None or not az.eligible or not bz.eligible:
            continue
        cube = (b.cube_rms_mm - bz.cube_rms_mm) - (a.cube_rms_mm - az.cube_rms_mm)
        pinch = (b.pinch_rms_mm - bz.pinch_rms_mm) - (a.pinch_rms_mm - az.pinch_rms_mm)
        result.append({"seed": a.seed, "mass_kg": a.mass_kg, "pulse": a.pulse,
                       "adjusted_cube_difference_mm": cube, "adjusted_pinch_difference_mm": pinch})
    return result


def assess(rows: list[DisturbanceOutcome], protocol: RobustnessProtocol) -> dict:
    """Require practical benefit and no new paired failures or safety regressions."""
    all_pairs = matched(rows)
    pairs = [(a, b) for a, b in all_pairs if a.pulse != "none" and a.eligible and b.eligible]
    extra = sum(a.success and not b.success for a, b in pairs)
    retention = interval([(a.seed, float(b.success) - float(a.success)) for a, b in pairs], protocol)
    precision = precision_pairs(rows, pairs)
    cube = interval([(p["seed"], p["adjusted_cube_difference_mm"]) for p in precision], protocol)
    pinch = interval([(p["seed"], p["adjusted_pinch_difference_mm"]) for p in precision], protocol)
    safe = not any(r.table_steps or r.warning or r.approach_contact for r in rows if r.controller == "sac")
    no_new_failures = not any(a.success and not b.success for a, b in all_pairs)
    retention_gain = retention["clusters"] > 1 and retention["ci95"][0] > 0
    precision_gain = (cube["clusters"] > 1 and cube["ci95"][1] < -protocol.practical_cube_improvement_mm
                      and pinch["ci95"][1] <= protocol.permitted_pinch_increase_mm)
    return {"common_eligible_disturbed_pairs": len(pairs), "additional_rl_failures": extra,
            "retention_not_worse": not bool(extra), "no_new_overall_paired_failures": no_new_failures,
            "overall_final_successes": {c: sum(r.success for r in rows if r.controller == c) for c in ("scripted", "sac")},
            "retention_rl_minus_script": retention, "adjusted_cube_difference_mm": cube,
            "adjusted_pinch_difference_mm": pinch, "safe": safe,
            "useful_benefit_established": bool(safe and no_new_failures and (retention_gain or precision_gain)),
            "start_full_training": False}


def fingerprints(repo: CartesianZeroRepository) -> dict[str, str]:
    """Protect old artifacts plus v3 weights, normalization and runtime source files."""
    result = protected_inputs()
    files = [repo.path, repo.root / "seed_0/model.zip", repo.root / "seed_0/vecnormalize.pkl"]
    files += list(Path("src/rbe_501_so101").rglob("*.py"))
    result.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return result


def run(repo: CartesianZeroRepository, output: Path) -> None:
    """Save the protocol first and verify all frozen sources/artifacts after evaluation."""
    output.mkdir(parents=True, exist_ok=False)
    protocol, hashes = RobustnessProtocol(), fingerprints(repo)
    write(output / "protocol.json", {"protocol": asdict(protocol), "preset": asdict(repo.load()),
        "pulses": [asdict(p) for p in pulses(protocol)], "input_sha256": hashes,
        "primary": "Paired final retention after common eligible holds; placement-cluster bootstrap.",
        "benefit_rule": "Positive retention difference CI or >0.05mm adjusted cube RMS improvement CI with <=0.05mm pinch increase CI; no new paired failures/contacts/warnings.",
        "mode": "Training-domain fixed nominal arm effects; one frozen trained seed; no updates."})
    rows = sweep(repo, output, protocol)
    pairs = [(a, b) for a, b in matched(rows) if a.pulse != "none" and a.eligible and b.eligible]
    write(output / "paired_motion.json", precision_pairs(rows, pairs))
    write(output / "assessment.json", assess(rows, protocol))
    assert hashes == fingerprints(repo), "A frozen input changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(CartesianZeroRepository(args.model_dir), args.out)

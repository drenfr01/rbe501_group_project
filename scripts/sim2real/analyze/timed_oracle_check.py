"""Frozen timing repair: recorded regressions and one independent paired validation."""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from rbe_501_so101.control.timed_oracle_acquisition import TimingConfig, make_timed_env
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository, make_env, snapshot
from scripts.sim2real.analyze.calibrated_lift_pilot import ZeroResidual, write
from scripts.sim2real.analyze.cartesian_disturbance import fingerprints
from scripts.sim2real.analyze.oracle_acquisition_check import OracleProtocol, assess, extra_diagnostics
from scripts.sim2real.analyze.pick_lift_coverage import CoverageProtocol, DiagnosticPlacer, PoseCase, cases, measure


@dataclass(frozen=True)
class TimedProtocol:
    """One timing configuration and disjoint recorded/fresh placement sets."""

    timing: TimingConfig = TimingConfig()
    recorded: CoverageProtocol = replace(CoverageProtocol(), first_yaw_seed=85000)
    validation: CoverageProtocol = replace(CoverageProtocol(), radii_m=(), first_yaw_seed=86000)


def evaluate(repo, protocol: CoverageProtocol, timing: TimingConfig, output: Path, controllers) -> list[dict]:
    """Change only the candidate phase timing; compare against the frozen original script."""
    output.mkdir(parents=True, exist_ok=False)
    preset, rows = repo.load(), []
    for name in controllers:
        for mass in protocol.masses_kg:
            base = replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass))
            cfg = replace(preset, base=base)
            env = make_timed_env(cfg, timing) if name == "timed" else make_env(cfg)
            raw = env.unwrapped
            raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z, PoseCase("initial"))
            try:
                for kind in ("workspace", "yaw"):
                    group = [(c, s) for c, s in cases(protocol) if (c.radius_m is not None) == (kind == "workspace")]
                    recent = []
                    for case, seed in group:
                        record = measure(env, ZeroResidual(), name, mass, case, seed)
                        recent.append({**asdict(record), "diagnostics": extra_diagnostics(env),
                                       "final_phase": str(env._controller.phase), "real_steps": env._controller._step})
                    rows.extend(recent)
                    write(output / "episodes.json", rows)
                    print(output.name, name, mass, kind, sum(r["task"]["final_success"] for r in recent),
                          "/", len(recent), "table", sum(r["task"]["robot_table_contact_steps"] > 0 for r in recent), flush=True)
            finally:
                env.close()
    return rows


def decision(recorded: list[dict], validation: list[dict], protocol: TimedProtocol) -> dict:
    """Apply the earlier held-out gate and require all recorded regressions to pass."""
    renamed = [{**r, "controller": "oracle" if r["controller"] == "timed" else r["controller"]} for r in validation]
    result = assess(renamed, OracleProtocol(validation=protocol.validation))
    result["successes"]["timed"] = result["successes"].pop("oracle")
    recorded_ok = len(recorded) == len(cases(protocol.recorded)) * len(protocol.recorded.masses_kg)
    recorded_ok &= all(r["task"]["final_success"] for r in recorded)
    safe_recorded = not any(r["task"]["robot_table_contact_steps"] or r["task"]["approach_contact_steps"] or r["task"]["warnings"] for r in recorded)
    result.update(recorded_all_pass=bool(recorded_ok), recorded_safe=safe_recorded,
                  passed_combined_gate=bool(result["passed_validation"] and recorded_ok and safe_recorded),
                  start_training=False, scope="Scripted oracle/timing validation, not evidence of RL improvement or hardware readiness.")
    return result


def run(repo: CartesianZeroRepository, output: Path) -> None:
    """Freeze the protocol and all runtime inputs before recorded or independent testing."""
    output.mkdir(parents=True, exist_ok=False)
    protocol, hashes = TimedProtocol(), fingerprints(repo)
    previous = Path("experiments/2026-11-26_oracle_acquisition")
    original = json.loads((previous / "input_verification.json").read_text())["sha256"]
    assert all(hashes.get(k) == v for k, v in original.items()), "A prior baseline or oracle input changed"
    write(output / "protocol.json", {"protocol": asdict(protocol), "snapshot": snapshot(repo.load()),
        "input_sha256": hashes, "gate": "Earlier independent gate plus every recorded grid/yaw case succeeds with zero candidate sampled table/approach contacts or warnings.",
        "scope": "Only phase timing and guard synchronization; original 200 steps, stable alignment/contact thresholds and closing dwell retained.",
        "bring_up": "Recorded radius .20m azimuth -40deg 10g seed83000 repairs with unchanged guards; no held-out tuning."})
    recorded = evaluate(repo, protocol.recorded, protocol.timing, output / "recorded", ("timed",))
    validation = evaluate(repo, protocol.validation, protocol.timing, output / "validation", ("scripted", "timed"))
    write(output / "assessment.json", decision(recorded, validation, protocol))
    assert hashes == fingerprints(repo), "Frozen inputs changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(CartesianZeroRepository(args.model_dir), args.out)

"""One frozen oracle prototype and one independent yaw validation; no RL training."""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.control.oracle_acquisition import make_oracle_env
from rbe_501_so101.kinematics.oracle_jaw_ik import JawIKConfig
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository, make_env, snapshot
from scripts.sim2real.analyze.calibrated_lift_pilot import ZeroResidual, write
from scripts.sim2real.analyze.cartesian_disturbance import fingerprints, interval, RobustnessProtocol
from scripts.sim2real.analyze.pick_lift_coverage import CoverageProtocol, DiagnosticPlacer, PoseCase, cases, measure


@dataclass(frozen=True)
class OracleProtocol:
    """Fixed development and validation sets; recorded seed 82000 was used during prototyping."""

    jaw: JawIKConfig = JawIKConfig()
    development: CoverageProtocol = CoverageProtocol()
    validation: CoverageProtocol = replace(CoverageProtocol(), radii_m=(), first_yaw_seed=85000)
    minimum_success_per_yaw: float = .95


def extra_diagnostics(env) -> dict:
    """Capture measured closure readiness and the last acquisition IK residual separately."""
    controller = env._controller
    solver = controller._solver
    result = getattr(solver, "last_result", None)
    error = controller.alignment_error
    jaw_error = getattr(controller, "jaw_alignment_rad", None)
    return {"closure_position_error_mm": float(np.linalg.norm(error) * 1000) if error is not None else None,
            "closure_axis_error_deg": float(np.rad2deg(jaw_error)) if jaw_error is not None else None,
            "closure_tilt_deg": float(np.rad2deg(controller.alignment_tilt)) if controller.alignment_tilt is not None else None,
            "ik_position_error_mm": result.error_m * 1000 if result else None,
            "ik_axis_error_deg": float(np.rad2deg(result.axis_error_rad)) if result else None,
            "close_start_step": controller.close_start_step}


def evaluate(repo, protocol: CoverageProtocol, jaw: JawIKConfig, output: Path, controllers) -> list[dict]:
    """Pair original and oracle acquisition without loading or modifying neural policies."""
    output.mkdir(parents=True, exist_ok=False)
    preset, rows = repo.load(), []
    for name in controllers:
        for mass in protocol.masses_kg:
            base = replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass))
            cfg = replace(preset, base=base)
            env = make_oracle_env(cfg, jaw) if name == "oracle" else make_env(cfg)
            raw = env.unwrapped
            raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z, PoseCase("initial"))
            try:
                for kind in ("workspace", "yaw"):
                    group = [(c, s) for c, s in cases(protocol) if (c.radius_m is not None) == (kind == "workspace")]
                    recent = []
                    for case, seed in group:
                        record = measure(env, ZeroResidual(), name, mass, case, seed)
                        recent.append({**asdict(record), "diagnostics": extra_diagnostics(env)})
                    rows.extend(recent)
                    write(output / "episodes.json", rows)
                    print(output.name, name, mass, kind, sum(r["task"]["final_success"] for r in recent),
                          "/", len(recent), "table", sum(r["task"]["robot_table_contact_steps"] > 0 for r in recent), flush=True)
            finally:
                env.close()
    return rows


def keyed(rows, name):
    """Key paired outcomes by actual mass, placement and orientation condition."""
    return {(r["mass_kg"], r["case"]["name"], r["task"]["placement_seed"]): r
            for r in rows if r["controller"] == name}


def assess(rows: list[dict], protocol: OracleProtocol) -> dict:
    """Do not promote a candidate that trades nominal success or contact safety for rotation gains."""
    baseline, oracle = keyed(rows, "scripted"), keyed(rows, "oracle")
    expected = len(protocol.validation.yaw_offsets_deg) * len(protocol.validation.masses_kg) * protocol.validation.yaw_placements
    complete = baseline.keys() == oracle.keys() and len(baseline) == expected
    failures = [k for k in baseline.keys() & oracle.keys() if baseline[k]["task"]["final_success"] and not oracle[k]["task"]["final_success"]]
    per_yaw = []
    for angle in protocol.validation.yaw_offsets_deg:
        a = [r for r in oracle.values() if r["case"]["yaw_offset_deg"] == angle]
        per_yaw.append({"angle_deg": angle, "episodes": len(a), "successes": sum(r["task"]["final_success"] for r in a)})
    safe = not any(r["task"]["robot_table_contact_steps"] or r["task"]["approach_contact_steps"] or r["task"]["warnings"] for r in oracle.values())
    nominal_ok = all(r["task"]["final_success"] for r in oracle.values() if r["case"]["yaw_offset_deg"] == 0)
    yaw_ok = all(r["episodes"] and r["successes"] / r["episodes"] >= protocol.minimum_success_per_yaw for r in per_yaw)
    diffs = [(a["task"]["placement_seed"], float(oracle[k]["task"]["final_success"]) - float(a["task"]["final_success"]))
             for k, a in baseline.items() if k in oracle and a["case"]["yaw_offset_deg"] != 0]
    gain = interval(diffs, RobustnessProtocol())
    better = gain["clusters"] > 1 and gain["ci95"][0] > 0
    return {"complete_pairs": complete, "per_yaw": per_yaw,
            "successes": {n: sum(r["task"]["final_success"] for r in rows if r["controller"] == n) for n in ("scripted", "oracle")},
            "new_paired_failures": [list(k) for k in failures], "safe": safe, "nominal_pass": nominal_ok,
            "rotated_success_gain": gain, "passed_validation": bool(complete and safe and nominal_ok and yaw_ok and better and not failures),
            "start_training": False}


def run(repo: CartesianZeroRepository, output: Path) -> None:
    """Freeze both stages before evaluation and refuse edits to protected inputs."""
    output.mkdir(parents=True, exist_ok=False)
    protocol, hashes = OracleProtocol(), fingerprints(repo)
    historical = Path("experiments/2026-11-26_pick_lift_coverage")
    original = json.loads((historical / "input_verification.json").read_text())["sha256"]
    assert all(hashes.get(k) == v for k, v in original.items()), "Original coverage baseline changed"
    write(output / "protocol.json", {"protocol": asdict(protocol), "snapshot": snapshot(repo.load()),
        "input_sha256": hashes, "historical_baseline": str(historical),
        "gate": "No new paired failures; all nominal controls pass; >=95% per yaw; positive rotated-success gain CI; zero candidate table/approach contacts and warnings.",
        "scope": "Oracle initial cube yaw only; fixed measured pose/tilt/jaw closure gate; no observations or saved policies changed.",
        "development_note": "Seed82000 10g yaw0/30/45 used for bring-up; 1deg solve versus 5deg measured closure avoids tracking-margin deadlock."})
    evaluate(repo, protocol.development, protocol.jaw, output / "development", ("oracle",))
    write(output / "development/reference.json", [r for r in json.loads((historical / "episodes.json").read_text()) if r["controller"] == "scripted"])
    validation = evaluate(repo, protocol.validation, protocol.jaw, output / "validation", ("scripted", "oracle"))
    write(output / "assessment.json", assess(validation, protocol))
    assert hashes == fingerprints(repo), "Frozen source, model or checkpoint changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(CartesianZeroRepository(args.model_dir), args.out)

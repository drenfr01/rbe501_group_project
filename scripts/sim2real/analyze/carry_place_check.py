"""Headless complete-task prototype: validated pickup, carry, supported release and retreat."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from rbe_501_so101.control.carry_place import CarryPlace, TransferConfig
from rbe_501_so101.control.contact_tracked_place import ContactTrackedCarryPlace
from rbe_501_so101.control.supported_place import SupportSearchConfig, SupportedCarryPlace
from rbe_501_so101.control.tracked_place import PlacementTrackingConfig, TrackedCarryPlace
from rbe_501_so101.control.timed_oracle_acquisition import make_timed_env
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.cartesian_disturbance import fingerprints
from scripts.sim2real.analyze.pick_lift_coverage import DiagnosticPlacer, PoseCase


@dataclass(frozen=True)
class PlacementProtocol:
    """A small first-task validation, separate from the old pick/lift horizon or RL rewards."""

    first_seed: int = 88000
    placements: int = 5
    masses_kg: tuple[float, ...] = (.005, .01, .015, .02)
    yaw_offsets_deg: tuple[float, ...] = (-30., 0., 30.)
    transfer: TransferConfig = TransferConfig()


def images(sequence: CarryPlace, output: Path) -> None:
    """Render recorded phase endpoints and a non-colliding magenta destination marker."""
    model = sequence.backend.model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = sequence.target + [0, 0, .04]
    camera.distance, camera.azimuth, camera.elevation = .48, 135, -25
    endpoints = {f.phase: f for f in sequence.frames}
    try:
        for phase, frame in endpoints.items():
            data.qpos[:] = frame.qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            scene = renderer.scene
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.full(3, .005), sequence.target, np.eye(3).ravel(), np.array([1., 0., 1., .6]))
            scene.ngeom += 1
            Image.fromarray(renderer.render()).save(output / f"{phase}.png")
    finally:
        renderer.close()


def watch(sequence: CarryPlace) -> None:
    """Play back the recorded physical qpos trajectory without running a second controller."""
    import time
    import mujoco.viewer

    model, data = sequence.backend.model, mujoco.MjData(sequence.backend.model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = sequence.target + [0, 0, .04]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .48, 135, -25
        for frame in sequence.frames:
            if not viewer.is_running():
                break
            data.qpos[:] = frame.qpos
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(sequence.raw.control_dt)


def episode(repo, cfg: TransferConfig, seed: int, mass: float, yaw: float, output: Path | None = None, viewer=False,
            sequence_type: type[CarryPlace] = CarryPlace) -> dict:
    """Record failure reasons as outcomes; do not continue motion after a failed guard."""
    preset = repo.load()
    preset = replace(preset, base=replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass)))
    env = make_timed_env(preset)
    raw = env.unwrapped
    raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z,
                                  PoseCase(f"yaw{yaw:g}", yaw_offset_deg=yaw))
    sequence = sequence_type(env, cfg)
    try:
        acquired = sequence.acquire(seed)
        success = acquired and sequence.run_transfer()
        last = sequence.frames[-1]
        result = {"seed": seed, "mass_kg": mass, "yaw_offset_deg": yaw, "acquisition_success": acquired,
            "complete_success": bool(success), "failure": sequence.failure, "destination_m": sequence.target.tolist(),
            "final_cube_m": list(last.cube_m), "final_xy_error_mm": float(np.linalg.norm(np.asarray(last.cube_m)[:2] - sequence.target[:2]) * 1000),
            "supported": last.supported, "grasped": last.grasped, "final_speed_m_s": last.speed_m_s,
            "table_contact_steps": sum(f.table_contact for f in sequence.frames),
            "warning_steps": sum(f.warning for f in sequence.frames), "phase_steps": dict(Counter(f.phase for f in sequence.frames))}
        if output is not None:
            output.mkdir(parents=True, exist_ok=False)
            write(output / "result.json", result)
            write(output / "trace.json", [asdict(f) for f in sequence.frames])
            images(sequence, output)
        if viewer:
            watch(sequence)
        return result
    finally:
        env.close()


def run(repo, output: Path, protocol: PlacementProtocol = PlacementProtocol(), sequence_type: type[CarryPlace] = CarryPlace) -> None:
    """Freeze one first-demo configuration and test 60 fresh complete sequences without training."""
    output.mkdir(parents=True, exist_ok=False)
    hashes = fingerprints(repo)
    write(output / "protocol.json", {"protocol": asdict(protocol), "input_sha256": hashes,
        "scope": "Unchanged 200-step acquisition then separate bounded backend carry/place; no Gym stepping past truncation or RL task changes.",
        "gate": "All acquisitions retained, >=95% complete per mass/yaw, zero candidate robot/table contacts and warnings; final supported released rest within 10mm; 40mm hand retreat.",
        "sequence_class": sequence_type.__name__,
        "support_search_m": SupportSearchConfig().maximum_descent_m if issubclass(sequence_type, SupportedCarryPlace) else 0.,
        "tracking": asdict(PlacementTrackingConfig()) if issubclass(sequence_type, TrackedCarryPlace) else None,
        "development": "87000 demo; support: 88002 5g yaw0; tracking: 89000 5g yaw+30; contact-aware: 88000 15g yaw-30 if enabled. No validation tuning."})
    write(output / "demo.json", episode(repo, protocol.transfer, 87000, .01, 0, output / "demo", sequence_type=sequence_type))
    rows = []
    for mass in protocol.masses_kg:
        for yaw in protocol.yaw_offsets_deg:
            recent = [episode(repo, protocol.transfer, seed, mass, yaw, sequence_type=sequence_type)
                      for seed in range(protocol.first_seed, protocol.first_seed + protocol.placements)]
            rows.extend(recent)
            write(output / "episodes.json", rows)
            print(mass, yaw, "complete", sum(r["complete_success"] for r in recent), "/", len(recent),
                  "failures", Counter(r["failure"] for r in recent if r["failure"]), flush=True)
    groups = [[r for r in rows if r["mass_kg"] == m and r["yaw_offset_deg"] == y]
              for m in protocol.masses_kg for y in protocol.yaw_offsets_deg]
    passed = all(r["acquisition_success"] and not r["table_contact_steps"] and not r["warning_steps"] for r in rows)
    passed &= all(sum(r["complete_success"] for r in group) / len(group) >= .95 for group in groups)
    write(output / "assessment.json", {"episodes": len(rows), "complete_successes": sum(r["complete_success"] for r in rows),
        "acquisition_successes": sum(r["acquisition_success"] for r in rows), "passed_gate": bool(passed),
        "failures": dict(Counter(r["failure"] for r in rows if r["failure"])), "start_training": False})
    assert hashes == fingerprints(repo)
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/calibrated_cartesian_zero_v3_pilot"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--supported", action="store_true", help="Use the bounded 3mm support-search candidate; validation seeds 89000-89004")
    selection.add_argument("--tracked", action="store_true", help="Use bounded final-lowering tracking correction; validation seeds 90000-90009")
    selection.add_argument("--contact-tracked", action="store_true", help="Freeze vertical tracking on support; validation seeds 91000-91009")
    parser.add_argument("--seed", type=int, default=87000)
    parser.add_argument("--mass-kg", type=float, default=.01)
    parser.add_argument("--yaw-offset-deg", type=float, default=0.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    repo = CartesianZeroRepository(args.model_dir)
    sequence_type = CarryPlace
    protocol = PlacementProtocol()
    if args.supported:
        sequence_type = SupportedCarryPlace
        protocol = replace(protocol, first_seed=89000)
    elif args.tracked:
        sequence_type = TrackedCarryPlace
        protocol = replace(protocol, first_seed=90000, placements=10)
    elif args.contact_tracked:
        sequence_type = ContactTrackedCarryPlace
        protocol = replace(protocol, first_seed=91000, placements=10)
    if args.replay:
        print(episode(repo, TransferConfig(), args.seed, args.mass_kg, args.yaw_offset_deg, args.out, args.viewer, sequence_type))
    else:
        run(repo, args.out, protocol, sequence_type)

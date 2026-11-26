"""Replay one exact lift case with contact/state traces; no controller changes.

    uv run --locked python -m scripts.sim2real.analyze.replay_pick_lift --controller random_5_20g --mass-kg 0.005 --placement-seed 20008 --calibrated --frames --out experiments/lift_drop_replay

Default: headless. --viewer shows the live episode in real time; --frames saves
offscreen phase snapshots. --calibrated means measured stiffness ONLY. Output
must be a new directory. Saved policies, normalizers, and physics stay fixed.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from rbe_501_so101.config import ScriptedPickLiftConfig
from scripts.sim2real.analyze.pick_lift_calibration import CONTROLLERS, build_condition, make_config

FRAME_STEPS = (39, 69, 99, 129, 199)
SIGNALS = ("grasped", "success", "table_contact")


def phase_at(step: int) -> str:
    """Match the existing zero-based scripted phase schedule."""
    cfg = ScriptedPickLiftConfig()
    if step < cfg.approach_steps:
        return "approach"
    if step < cfg.approach_steps + cfg.descend_steps:
        return "descend"
    if step < cfg.approach_steps + cfg.descend_steps + cfg.close_steps:
        return "close"
    return "lift"


def transition_events(frames: list[dict]) -> list[dict]:
    """Timestamp contact, success, and table-contact changes, including losses."""
    previous = dict.fromkeys(SIGNALS, False)
    events = []
    for frame in frames:
        for signal in SIGNALS:
            if frame[signal] != previous[signal]:
                events.append({"step": frame["step"], "signal": signal, "value": frame[signal]})
                previous[signal] = frame[signal]
    return events


def contact_records(raw) -> list[dict]:
    """Current solver contact positions, signed gaps, and normal-force magnitudes."""
    model, data = raw.backend.model, raw.backend.data
    result = []
    for index, contact in enumerate(data.contact):
        geoms = (int(contact.geom1), int(contact.geom2))
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        result.append({
            "bodies": [model.body(int(model.geom_bodyid[g])).name for g in geoms],
            "geoms": [model.geom(g).name or f"geom_{g}" for g in geoms],
            "position_m": contact.pos.tolist(), "gap_mm": float(contact.dist * 1000),
            "normal_force_n": float(abs(wrench[0])),
        })
    return result


def snapshot(raw, info: dict, step: int, action: np.ndarray) -> dict:
    """One control-boundary state, in world meters and joint degrees."""
    state = raw.grasp_state
    contacts = contact_records(raw)
    table = raw.config.sim.table_body
    table_contact = any(table in c["bodies"] and "cube" not in c["bodies"] for c in contacts)
    return {
        "step": step + 1, "time_s": float(raw.backend.data.time), "phase": phase_at(step),
        "grasped": bool(info["grasped"]), "success": bool(info["is_success"]),
        "lift_m": float(info["lift_m"]), "table_contact": table_contact,
        "pinch_m": state.p_pinch.tolist(), "cube_m": state.obj.p_s.tolist(),
        "cube_minus_pinch_m": (state.obj.p_s - state.p_pinch).tolist(),
        "joint_deg": np.rad2deg(state.robot.qpos).tolist(),
        "command_deg": np.rad2deg(state.robot.q_cmd).tolist(),
        "joint_error_deg": np.rad2deg(state.robot.q_cmd - state.robot.qpos).tolist(),
        "policy_action": action.tolist(), "contacts": contacts,
        "warnings": raw.backend.has_warnings(),
        "lift_released": info.get("lift_released"), "grasp_abort": info.get("grasp_abort"),
    }


def save_frame(renderer, raw, path: Path) -> None:
    """Render a close-up of the current grasp scene without affecting dynamics."""
    camera = mujoco.MjvCamera()
    camera.lookat[:] = raw.grasp_state.obj.p_s + [0, 0, 0.04]
    camera.distance, camera.azimuth, camera.elevation = 0.48, 135, -25
    renderer.update_scene(raw.backend.data, camera=camera)
    Image.fromarray(renderer.render()).save(path)


def rollout(env, policy, seed: int, output: Path, frames: bool, viewer: bool) -> list[dict]:
    """Replay the unchanged episode, optionally displaying it and saving images."""
    raw = env.unwrapped
    obs, _ = env.reset(seed=seed)
    policy.reset()
    renderer = mujoco.Renderer(raw.backend.model, 480, 640) if frames else None
    trace = []
    try:
        for step in range(raw.config.task.max_episode_steps):
            if viewer and not raw.viewer_open:
                raise RuntimeError("Viewer closed before replay finished")
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                raise RuntimeError("Nonfinite replay state")
            trace.append(snapshot(raw, info, step, action))
            if renderer is not None and step in FRAME_STEPS:
                save_frame(renderer, raw, output / f"step_{step+1:03d}_{phase_at(step)}.png")
            if viewer:
                time.sleep(raw.control_dt)
            if terminated or truncated:
                break
    finally:
        if renderer is not None:
            renderer.close()
    return trace


def replay(controller: str, mass: float, seed: int, calibrated: bool, output: Path,
           frames: bool = False, viewer: bool = False, guarded: bool = False,
           arm_effects: bool = False, clearance_m: float = 0.0) -> dict:
    """Write a new trace directory while leaving every input artifact unchanged."""
    output.mkdir(parents=True, exist_ok=False)
    config = make_config(mass, calibrated)
    if guarded or arm_effects or clearance_m:
        if not calibrated:
            raise ValueError("Experimental guard/clearance/arm-effect replays require --calibrated")
        from scripts.sim2real.analyze.validate_pick_lift_setup import Setup, build
        setup = Setup("replay", clearance_m, guarded, arm_effects)
        env, policy = build(setup, controller, mass, "human" if viewer else None)
    else:
        env, policy = build_condition(config, controller, "human" if viewer else None)
    try:
        trace = rollout(env, policy, seed, output, frames, viewer)
    finally:
        env.close()
        getattr(policy, "close", lambda: None)()
    result = {"controller": controller, "mass_kg": mass, "placement_seed": seed,
              "calibrated_stiffness_only": calibrated and not arm_effects,
              "guarded": guarded, "arm_effects": arm_effects, "clearance_m": clearance_m,
              "events": transition_events(trace), "trace": trace}
    (output / "trace.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{output}: final success={trace[-1]['success']}, lift={trace[-1]['lift_m']*100:.2f} cm", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", choices=CONTROLLERS, default="scripted")
    parser.add_argument("--mass-kg", type=float, default=0.01)
    parser.add_argument("--placement-seed", type=int, default=20000)
    parser.add_argument("--calibrated", action="store_true")
    parser.add_argument("--frames", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--guarded", action="store_true", help="Require stable grasp before lift; stop arm commands on loss")
    parser.add_argument("--arm-effects", action="store_true", help="Experimental measured arm servo/sensing effects")
    parser.add_argument("--clearance-mm", type=float, default=0.0, help="Experimental grasp waypoint offset; frozen candidate uses 0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.guarded or args.arm_effects or args.clearance_mm) and not args.calibrated:
        parser.error("--guarded, --arm-effects, and --clearance-mm require --calibrated")
    torch.set_num_threads(1)
    replay(args.controller, args.mass_kg, args.placement_seed, args.calibrated, args.out, args.frames, args.viewer,
           args.guarded, args.arm_effects, args.clearance_mm / 1000)

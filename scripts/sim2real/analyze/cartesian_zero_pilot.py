"""Fresh 10k CUDA SAC pilot on the validated Cartesian/zero-jaw scripted reference."""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from rbe_501_so101.config import PickLiftEvalConfig, TrainConfig
from rbe_501_so101.envs.model_indices import descendant_body_ids
from rbe_501_so101.rl.cartesian_zero_training import (
    CartesianZeroPreset, CartesianZeroRepository, factory, load_policy, make_env,
)
from rbe_501_so101.rl.training import train_with_factory
from scripts.sim2real.analyze.acquisition_holding_checks import displacement_rms, physical_state
from scripts.sim2real.analyze.calibrated_lift_pilot import (
    PilotConfig, ZeroResidual, artifact_hashes, training_history, write,
)
from scripts.sim2real.analyze.limited_lift_pilot import protected_inputs as previous_inputs
from scripts.sim2real.analyze.pick_lift_calibration import EpisodeResult, robot_touches_table, summarize
from scripts.sim2real.analyze.raised_approach_check import touches_cube

PILOT = replace(PilotConfig(), eval_seed=80_000)


@dataclass(frozen=True)
class PilotEpisode(EpisodeResult):
    """Add acquisition and physical tail-motion measurements to the common task outcome."""

    approach_contact_steps: int
    preclose_max_xy_mm: float
    cube_tail_rms_mm: float
    pinch_tail_rms_mm: float


def protected_inputs() -> dict[str, str]:
    """Protect both earlier pilots, the original policies and the validated controller."""
    result = previous_inputs()
    paths = [Path("checkpoints/calibrated_lift_only_v2_pilot") / p
             for p in ("run_config.json", "seed_0/model.zip", "seed_0/vecnormalize.pkl")]
    paths += [Path("src/rbe_501_so101/control") / p for p in ("cartesian_descent.py", "gripper_closing.py")]
    paths += [Path("experiments/2026-11-26_gripper_closing/frozen_selection.json")]
    result.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return result


def preflight(preset: CartesianZeroPreset) -> dict:
    """Check the actual v3 environment and real CUDA arithmetic before creating weights."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this pilot; no silent CPU fallback")
    product = torch.eye(16, device="cuda") @ torch.ones((16, 16), device="cuda")
    assert product.sum().item() == 256
    torch.cuda.synchronize()
    env = make_env(preset)
    try:
        check_env(env, warn=True)
    finally:
        env.close()
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "environment_checker": "passed", "cuda_arithmetic": "passed",
            "versions": {p: version(p) for p in ("mujoco", "gymnasium", "stable-baselines3", "numpy")}}


def episode(env, policy, seed: int) -> PilotEpisode:
    """Measure the full task, including acquisition that RL is not allowed to change."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    raw = env.unwrapped
    bodies = descendant_body_ids(raw.backend.model, raw.config.sim.robot_root_body)
    initial = physical_state(raw).obj.p_s.copy()
    table = approach = 0
    shift = max_lift = 0.
    ever_success = ever_grasped = warnings = False
    cubes, pinches = [], []
    for step in range(raw.config.task.max_episode_steps):
        obs, reward, _, _, info = env.step(policy(obs))
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError("Nonfinite v3 pilot rollout")
        state = physical_state(raw)
        table += int(robot_touches_table(raw, bodies))
        approach += int(step < 40 and touches_cube(raw, bodies))
        if not env._controller.close_started:
            shift = max(shift, float(np.linalg.norm((state.obj.p_s - initial)[:2]) * 1000))
        ever_success |= bool(info["is_success"])
        ever_grasped |= bool(info["grasped"])
        warnings |= raw.backend.has_warnings()
        max_lift = max(max_lift, float(info["lift_m"]))
        cubes.append(state.obj.p_s.copy())
        pinches.append(state.p_pinch.copy())
    cube, pinch = np.asarray(cubes[-30:]), np.asarray(pinches[-30:])
    return PilotEpisode(seed, bool(info["is_success"]), ever_success, ever_grasped,
                        float(info["lift_m"]), max_lift, table, warnings, approach, shift,
                        displacement_rms(cube, cube[0]) * 1000, displacement_rms(pinch, pinch[0]) * 1000)


def evaluate(preset: CartesianZeroPreset, policy, seeds, name: str) -> list[dict]:
    """Pair both dynamics modes and four masses, recording actual residual interventions."""
    rows = []
    for effects in (False, True):
        for mass in PickLiftEvalConfig().masses_kg:
            base = replace(preset.base, arm_effects=effects,
                           mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass))
            env = make_env(replace(preset, base=base))
            episodes, counters = [], []
            try:
                for seed in seeds:
                    episodes.append(episode(env, policy, seed))
                    counters.append(env.counters.copy())
                summary = summarize(preset.base.name, name, mass, episodes)
                rows.append({"summary": asdict(summary), "arm_effects": effects,
                             "episodes": [asdict(e) for e in episodes], "interventions": counters})
                print(name, effects, mass, f"success={summary.success_rate:.0%}",
                      "table", summary.robot_table_contact_episodes, flush=True)
            finally:
                env.close()
    return rows


def assess(reference: list[dict], learned: list[dict], known: list[dict]) -> dict:
    """Freeze reliability gates; a matched success count is not evidence of RL benefit."""
    def indexed(rows):
        """Use placement/mass/mode keys rather than pooling unmatched episodes."""
        return {(r["arm_effects"], r["summary"]["mass_kg"], e["placement_seed"]): e
                for r in rows for e in r["episodes"]}
    base, policy = indexed(reference), indexed(learned)
    enough = len(learned) == 8 and base.keys() == policy.keys() and bool(base)
    regressions = [key for key in base.keys() & policy.keys() if base[key]["final_success"] and not policy[key]["final_success"]]
    es = list(policy.values())
    strict = enough and all(r["summary"]["success_rate"] >= .95 for r in learned)
    strict = strict and not any(e["robot_table_contact_steps"] or e["approach_contact_steps"] or e["warnings"] for e in es)
    strict = strict and not any(e["ever_success"] and not e["final_success"] for e in es)
    known_ok = len(known) == 8 and all(e["final_success"] for r in known for e in r["episodes"])
    return {"passed_stability_gate": bool(strict and known_ok and not regressions),
            "known_regressions_pass": known_ok, "additional_failure_cases": [list(k) for k in regressions],
            "useful_benefit_established": False, "start_full_training": False,
            "reason": "A short stability pilot does not establish useful improvement over the scripted reference."}


def train(repo, preset, cfg) -> Path:
    """Fresh SAC seed 0 only; preserve the existing optimizer and network settings."""
    training = replace(TrainConfig(), timesteps=cfg.steps, checkpoint_freq_steps=cfg.steps // 2)
    repo.create(preset, training, cfg.seed, cfg.steps)
    seed_dir = repo.root / f"seed_{cfg.seed}"
    return train_with_factory(cfg.seed, factory(repo.load(), seed_dir / "train"), algo="sac",
                              timesteps=cfg.steps, seed_dir=seed_dir, cfg=training, device="cuda")


def run(output: Path, model_dir: Path) -> None:
    """Freeze a new protocol before any policy training or new-placement evaluation."""
    if model_dir.exists():
        raise FileExistsError("New model directory required")
    output.mkdir(parents=True, exist_ok=False)
    preset, hashes = CartesianZeroPreset(), protected_inputs()
    write(output / "protocol.json", {"pilot": asdict(PILOT), "preset": asdict(preset), "protected_sha256": hashes,
          "known_seeds": [68006, 69005, 75003], "heldout_seeds": [PILOT.eval_seed, PILOT.eval_seed + PILOT.episodes_per_mass - 1],
          "gate": ">=95% per mass/mode; no additional paired failures; known regressions pass; zero table/approach contact, warnings or lost achieved success"})
    write(output / "preflight.json", preflight(preset))
    reference = evaluate(preset, ZeroResidual(), range(PILOT.eval_seed, PILOT.eval_seed + PILOT.episodes_per_mass), "scripted")
    write(output / "scripted_eval.json", reference)
    repo = CartesianZeroRepository(model_dir)
    seed_dir = train(repo, preset, PILOT)
    write(output / "run_config.json", json.loads(repo.path.read_text()))
    finish(repo, output, seed_dir, reference)
    assert hashes == protected_inputs(), "A protected reference changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


def finish(repo, output, seed_dir, reference) -> None:
    """Reload saved policy/statistics, evaluate and preserve replay-buffer provenance."""
    policy = load_policy(repo, PILOT.seed)
    try:
        learned = evaluate(repo.load(), policy, range(PILOT.eval_seed, PILOT.eval_seed + PILOT.episodes_per_mass), "fresh_v3_sac")
        known = evaluate(repo.load(), policy, (68006, 69005, 75003), "known_v3_sac")
    finally:
        policy.close()
    write(output / "learned_eval.json", learned)
    write(output / "known_eval.json", known)
    write(output / "assessment.json", assess(reference, learned, known))
    write(output / "training_history.json", training_history(seed_dir))
    model = SAC.load(seed_dir / "model.zip", device="cuda")
    write(output / "artifacts.json", {"seed_dir": seed_dir, "sha256": artifact_hashes(seed_dir),
          "steps": model.num_timesteps, "device": str(model.device), "learning_starts": model.learning_starts,
          "buffer_size": model.buffer_size, "batch_size": model.batch_size, "gradient_steps": model.gradient_steps,
          "train_freq": str(model.train_freq), "policy": str(model.policy)})


def watch(repo, output: Path, mass: float, seed: int, viewer: bool) -> None:
    """Replay v3 using its matching controller, limits and frozen normalization statistics."""
    from scripts.sim2real.analyze.replay_pick_lift import rollout, transition_events

    preset = repo.load()
    base = replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass))
    env = make_env(replace(preset, base=base), "human" if viewer else None)
    policy = load_policy(repo)
    output.mkdir(parents=True, exist_ok=False)
    try:
        trace = rollout(env, policy, seed, output, frames=True, viewer=viewer)
        write(output / "trace.json", {"preset": asdict(preset), "mass_kg": mass, "seed": seed,
              "events": transition_events(trace), "trace": trace, "interventions": env.counters})
        print("V3 replay final success", trace[-1]["success"], flush=True)
    finally:
        env.close()
        policy.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--mass-kg", type=float, default=.005)
    parser.add_argument("--placement-seed", type=int, default=80000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.watch:
        watch(CartesianZeroRepository(args.model_dir), args.out, args.mass_kg, args.placement_seed, args.viewer)
    else:
        run(args.out, args.model_dir)

"""Fresh 10k CUDA SAC pilot for the versioned complete-task residual environment."""

import argparse
import json
import platform
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from rbe_501_so101.config import TrainConfig
from rbe_501_so101.envs import PICK_PLACE_ENV_ID
from rbe_501_so101.envs.pick_place_config import PickPlaceConfig
from rbe_501_so101.envs.pick_place_env import SO101PickPlaceEnv
from rbe_501_so101.rl.pick_place_training import PickPlaceRepository, factory, load_policy, snapshot
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository
from rbe_501_so101.rl.training import train_with_factory
from scripts.sim2real.analyze.calibrated_lift_pilot import artifact_hashes, training_history, write
from scripts.sim2real.analyze.cartesian_disturbance import fingerprints


@dataclass(frozen=True)
class FullTaskPilot:
    """One fresh training seed, short budget and fixed paired evaluation before promotion."""

    steps: int = 10_000
    seed: int = 0
    eval_first_seed: int = 93000
    placements: int = 5
    masses_kg: tuple[float, ...] = (.005, .01, .015, .02)
    yaw_offsets_deg: tuple[float, ...] = (-30., 0., 30.)


class ZeroPolicy:
    """The validated full-task scripted controller with no learned correction."""

    def reset(self) -> None:
        """No recurrent state to reset."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Zero residual means exact scripted baseline behavior."""
        return np.zeros(3, dtype=np.float32)


def episode(env, policy, seed: int, yaw: float) -> dict:
    """Measure only genuine full-task terminal success, retaining every failure and truncation."""
    obs, _ = env.reset(seed=seed, options={"yaw_offset_deg": yaw})
    policy.reset()
    for _ in range(env.config.max_episode_steps):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError("Nonfinite full-task rollout")
        if terminated or truncated:
            break
    return {"seed": seed, "yaw_offset_deg": yaw, "mass_kg": info["mass_kg"],
            "success": bool(info["is_success"]), "terminated": terminated, "truncated": truncated,
            "steps": info["episode_steps"], "error_mm": info["placement_error_m"] * 1000,
            "table_steps": info["robot_table_contact_steps"], "warning": info["mujoco_warning"],
            "residual_steps": info["residual_steps"], "residual_rejections": info["residual_rejections"],
            "failure_reason": info["failure_reason"], "released": info["released"], "grasped": info["grasped"]}


def evaluate(cfg: PickPlaceConfig, policy, pilot: FullTaskPilot, output: Path, label: str) -> list[dict]:
    """Pair all masses, orientations and placement seeds under one frozen task definition."""
    rows = []
    for mass in pilot.masses_kg:
        base = replace(cfg.preset.base, mass=replace(cfg.preset.base.mass, mode="fixed", fixed_kg=mass))
        fixed = replace(cfg, preset=replace(cfg.preset, base=base))
        env = SO101PickPlaceEnv(fixed)
        try:
            for yaw in pilot.yaw_offsets_deg:
                group = [episode(env, policy, seed, yaw)
                         for seed in range(pilot.eval_first_seed, pilot.eval_first_seed + pilot.placements)]
                rows.extend(group)
                write(output / f"{label}.json", rows)
                print(label, mass, yaw, sum(r["success"] for r in group), "/", len(group), flush=True)
        finally:
            env.close()
    return rows


def assess(reference: list[dict], learned: list[dict], pilot: FullTaskPilot) -> dict:
    """Matching scripted stability does not establish added benefit from a learned residual."""
    def keyed(rows):
        """Use true paired conditions rather than pooled success counts."""
        return {(r["seed"], r["mass_kg"], r["yaw_offset_deg"]): r for r in rows}
    baseline, candidate = keyed(reference), keyed(learned)
    expected = pilot.placements * len(pilot.masses_kg) * len(pilot.yaw_offsets_deg)
    complete = baseline.keys() == candidate.keys() and len(candidate) == expected
    regressions = [list(k) for k in baseline.keys() & candidate.keys() if baseline[k]["success"] and not candidate[k]["success"]]
    groups = [[r for r in learned if r["mass_kg"] == mass and r["yaw_offset_deg"] == yaw]
              for mass in pilot.masses_kg for yaw in pilot.yaw_offsets_deg]
    stable = complete and all(g and sum(r["success"] for r in g) / len(g) >= .95 for g in groups)
    stable &= not any(r["table_steps"] or r["warning"] for r in learned)
    return {"paired_conditions_complete": complete, "scripted_successes": sum(r["success"] for r in reference),
            "learned_successes": sum(r["success"] for r in learned), "episodes_each": expected,
            "new_failures": regressions, "passed_stability_gate": bool(stable and not regressions),
            "useful_rl_benefit_established": False, "start_long_training": False}


def preflight(cfg: PickPlaceConfig) -> dict:
    """Check the actual registered full-task environment and real CUDA arithmetic."""
    if not torch.cuda.is_available():
        raise RuntimeError("This short pilot requires CUDA; no silent CPU substitution")
    product = torch.eye(16, device="cuda") @ torch.ones((16, 16), device="cuda")
    assert product.sum().item() == 256
    env = SO101PickPlaceEnv(cfg)
    try:
        check_env(env, warn=True)
        gym_check(env, skip_render_check=True)
    finally:
        env.close()
    registered = gym.make(PICK_PLACE_ENV_ID, cfg=cfg)
    try:
        obs, _ = registered.reset(seed=92000)
        registered.step(np.zeros(3))
    finally:
        registered.close()
    return {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "cuda_arithmetic": "passed", "sb3_checker": "passed",
            "gymnasium_checker": "passed", "registered_env": PICK_PLACE_ENV_ID,
            "observation_shape": list(obs.shape), "action_shape": [3],
            "versions": {p: version(p) for p in ("mujoco", "gymnasium", "stable-baselines3", "numpy")}}


def run(output: Path, model_dir: Path) -> None:
    """Write the protocol before evaluating or creating entirely fresh policy weights."""
    if model_dir.exists():
        raise FileExistsError("New full-task model directory required")
    output.mkdir(parents=True, exist_ok=False)
    cfg, pilot = PickPlaceConfig(), FullTaskPilot()
    legacy = CartesianZeroRepository(Path("checkpoints/calibrated_cartesian_zero_v3_pilot"))
    hashes = fingerprints(legacy)
    write(output / "protocol.json", {"pilot": asdict(pilot), "snapshot": snapshot(cfg),
          "protected_sha256": hashes, "gate": ">=95% every mass/yaw; zero policy table contacts/warnings; no new paired failures",
          "scope": "Fresh 10k CUDA full-task residual stability pilot; no promotion or long training."})
    write(output / "preflight.json", preflight(cfg))
    reference = evaluate(cfg, ZeroPolicy(), pilot, output, "scripted_eval")
    if not all(r["success"] and not r["table_steps"] and not r["warning"] for r in reference):
        raise RuntimeError("Scripted full-task baseline failed; do not train as a substitute")
    train_and_evaluate(cfg, pilot, model_dir, output, reference)
    assert hashes == fingerprints(legacy), "A protected legacy artifact changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


def train_and_evaluate(cfg, pilot, model_dir, output, reference) -> None:
    """Reload the matching policy and normalizer before independent deterministic evaluation."""
    training = replace(TrainConfig(), timesteps=pilot.steps, checkpoint_freq_steps=pilot.steps // 2,
                       model_dir=model_dir, seeds=(pilot.seed,), eval_seed_offset=pilot.eval_first_seed,
                       eval_episodes=pilot.placements)
    repo = PickPlaceRepository(model_dir)
    repo.create(cfg, training, pilot.seed)
    folder = model_dir / f"seed_{pilot.seed}"
    train_with_factory(pilot.seed, factory(repo.load(), folder / "train"), algo="sac",
                       timesteps=pilot.steps, seed_dir=folder, cfg=training, device="cuda")
    write(output / "run_config.json", json.loads(repo.path.read_text()))
    policy = load_policy(repo)
    try:
        learned = evaluate(repo.load(), policy, pilot, output, "learned_eval")
    finally:
        policy.close()
    write(output / "assessment.json", assess(reference, learned, pilot))
    write(output / "training_history.json", training_history(folder))
    model = SAC.load(folder / "model.zip", device="cuda")
    write(output / "artifacts.json", {"steps": model.num_timesteps, "device": str(model.device),
          "sha256": artifact_hashes(folder), "policy": str(model.policy), "seed_dir": str(folder)})


def replay(repo, output: Path, seed: int, mass: float, yaw: float, scripted: bool, viewer: bool) -> None:
    """Save the actual full-task policy trajectory and optionally play it in the existing viewer."""
    from scripts.sim2real.analyze.carry_place_check import images, watch
    cfg = repo.load()
    base = replace(cfg.preset.base, mass=replace(cfg.preset.base.mass, mode="fixed", fixed_kg=mass))
    env = SO101PickPlaceEnv(replace(cfg, preset=replace(cfg.preset, base=base)))
    policy = ZeroPolicy() if scripted else load_policy(repo)
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = episode(env, policy, seed, yaw)
        write(output / "result.json", result)
        write(output / "trace.json", [asdict(f) for f in env.sequence.frames])
        images(env.sequence, output)
        if viewer:
            watch(env.sequence)
        print(result, flush=True)
    finally:
        env.close()
        if not scripted:
            policy.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/pick_place_v1_pilot"))
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--seed", type=int, default=93000)
    parser.add_argument("--mass-kg", type=float, default=.01)
    parser.add_argument("--yaw-offset-deg", type=float, default=0.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.replay:
        replay(PickPlaceRepository(args.model_dir), args.out, args.seed, args.mass_kg,
               args.yaw_offset_deg, args.scripted, args.viewer)
    else:
        run(args.out, args.model_dir)

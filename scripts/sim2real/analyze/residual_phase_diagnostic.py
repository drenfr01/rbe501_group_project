"""Ablate the failed pilot by phase, with frozen old weights and known placements."""

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.config import PickLiftEvalConfig
from rbe_501_so101.rl.calibrated_pick_lift import LiftPresetRepository, load_policy, make_env
from scripts.sim2real.analyze.calibrated_lift_pilot import write
from scripts.sim2real.analyze.replay_pick_lift import phase_at


@dataclass(frozen=True)
class PhaseMask:
    """Change only which residual components are applied, not the frozen network."""

    name: str
    phases: tuple[str, ...]
    multiplier: float = 1.


MASKS = (PhaseMask("all", ("approach", "descend", "close", "lift")),
         *(PhaseMask(p, (p,)) for p in ("approach", "descend", "close", "lift")),
         PhaseMask("none", ()), PhaseMask("small_lift", ("lift",), .1))


def episode(env, policy, seed: int, mask: PhaseMask) -> dict:
    """Measure phase-local contacts, success losses and residual magnitudes."""
    obs, _ = env.reset(seed=seed)
    phases = {p: {"steps": 0, "table_steps": 0, "success_losses": 0,
                   "action_norm_sum": 0.} for p in ("approach", "descend", "close", "lift")}
    previous_success, previous_contacts = False, 0
    for step in range(env.unwrapped.config.task.max_episode_steps):
        phase = phase_at(step)
        action = policy(obs) * mask.multiplier if phase in mask.phases else np.zeros(6)
        obs, reward, _, truncated, info = env.step(action)
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError("Nonfinite phase diagnostic state")
        row = phases[phase]
        row["steps"] += 1
        row["action_norm_sum"] += float(np.linalg.norm(action))
        row["table_steps"] += info["robot_table_contact_steps"] - previous_contacts
        row["success_losses"] += int(previous_success and not info["is_success"])
        previous_success, previous_contacts = info["is_success"], info["robot_table_contact_steps"]
        if truncated:
            break
    return {"seed": seed, "success": bool(info["is_success"]), "phases": phases,
            "table_steps": previous_contacts, "warning": info["mujoco_warning"]}


def run(repo: LiftPresetRepository, output: Path) -> None:
    """Known old eval placements are diagnostic only, never the new pilot holdout."""
    output.mkdir(parents=True, exist_ok=False)
    preset, policy = repo.load(), load_policy(repo)
    records = []
    try:
        for mask in MASKS:
            episodes = []
            for mass in PickLiftEvalConfig().masses_kg:
                fixed = replace(preset, mass=replace(preset.mass, mode="fixed", fixed_kg=mass))
                env = make_env(fixed)
                try:
                    episodes += [{"mass_kg": mass, **episode(env, policy, 66000 + n, mask)} for n in range(5)]
                finally:
                    env.close()
            records.append({"mask": asdict(mask), "episodes": episodes})
            write(output / "phase_ablation.json", records)
            print(mask.name, "success", sum(e["success"] for e in episodes),
                  "table", sum(e["table_steps"] > 0 for e in episodes), flush=True)
    finally:
        policy.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(LiftPresetRepository(args.model_dir), args.out)

"""Frozen workspace coverage and explicitly out-of-distribution cube-yaw probes."""

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rbe_501_so101.envs.protocols import ObjectPlacer
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroRepository, load_policy, make_env
from scripts.sim2real.analyze.calibrated_lift_pilot import ZeroResidual, write
from scripts.sim2real.analyze.cartesian_disturbance import fingerprints
from scripts.sim2real.analyze.cartesian_zero_pilot import PilotEpisode, episode


@dataclass(frozen=True)
class PoseCase:
    """Fixed polar position or normal sampled position, with an explicit yaw offset."""

    name: str
    radius_m: float | None = None
    azimuth_deg: float = 0.
    yaw_offset_deg: float = 0.


@dataclass(frozen=True)
class CoverageProtocol:
    """Evaluation-only matrix frozen before looking at coverage outcomes."""

    radii_m: tuple[float, ...] = (.14, .16, .18, .20, .22)
    azimuths_deg: tuple[float, ...] = (-40., -30., -20., -10., 0., 10., 20., 30., 40.)
    yaw_offsets_deg: tuple[float, ...] = (-45., -30., -15., 0., 15., 30., 45.)
    masses_kg: tuple[float, ...] = (.005, .01, .015, .02)
    grid_seed: int = 83_000
    first_yaw_seed: int = 82_000
    yaw_placements: int = 10


class DiagnosticPlacer:
    """Inject only the explicitly recorded initial pose without changing robot physics."""

    def __init__(self, base: ObjectPlacer, axis_xy: np.ndarray, rest_z: float, case: PoseCase) -> None:
        """Keep normal sampling available to match centers across yaw conditions."""
        self.base, self.axis_xy, self.rest_z, self.case = base, axis_xy, rest_z, case
        self.last_position = np.zeros(3)
        self.last_yaw = 0.

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        """Consume the normal placement draws even when using an explicit grid pose."""
        position, yaw = self.base.sample(rng)
        if self.case.radius_m is not None:
            yaw = float(np.deg2rad(self.case.azimuth_deg))
            xy = self.axis_xy + self.case.radius_m * np.array([np.cos(yaw), np.sin(yaw)])
            position = np.array([*xy, self.rest_z])
        yaw += float(np.deg2rad(self.case.yaw_offset_deg))
        self.last_position, self.last_yaw = position.copy(), yaw
        return position, yaw


@dataclass(frozen=True)
class CoverageOutcome:
    """Separate acquisition/guard outcomes from success under the original task."""

    controller: str
    mass_kg: float
    case: PoseCase
    initial_center_m: tuple[float, float, float]
    initial_yaw_rad: float
    task: PilotEpisode
    close_started: bool
    acquisition_abort: str | None
    lift_released: bool
    gate_aborted: bool
    ik_misses: int
    clearance_rejections: int


def cases(protocol: CoverageProtocol) -> list[tuple[PoseCase, int]]:
    """Cover workspace boundaries separately from unseen independent yaw rotations."""
    grid = [(PoseCase(f"r{r:.2f}_a{a:g}", r, a), protocol.grid_seed)
            for r in protocol.radii_m for a in protocol.azimuths_deg]
    rotated = [(PoseCase(f"yaw{yaw:g}", yaw_offset_deg=yaw), seed)
               for yaw in protocol.yaw_offsets_deg
               for seed in range(protocol.first_yaw_seed, protocol.first_yaw_seed + protocol.yaw_placements)]
    return grid + rotated


def measure(env, policy, controller: str, mass: float, case: PoseCase, seed: int) -> CoverageOutcome:
    """Use the pilot's unchanged 200-step outcome and physical-motion measurements."""
    placer = env.unwrapped._placer
    placer.case = case
    task = episode(env, policy, seed)
    script, gate = env._controller, env.env.env.gate
    return CoverageOutcome(controller, mass, case, tuple(placer.last_position), placer.last_yaw,
                           task, script.close_started, script.abort_reason, gate.released,
                           gate.aborted, script.ik_misses, script.clearance_rejections)


def sweep(repo: CartesianZeroRepository, output: Path, protocol: CoverageProtocol) -> list[CoverageOutcome]:
    """Change only initial object placement; preserve both frozen control pipelines."""
    from dataclasses import replace

    preset, learned = repo.load(), load_policy(repo)
    rows = []
    try:
        for name, policy in (("scripted", ZeroResidual()), ("sac", learned)):
            for mass in protocol.masses_kg:
                base = replace(preset.base, mass=replace(preset.base.mass, mode="fixed", fixed_kg=mass))
                env = make_env(replace(preset, base=base))
                raw = env.unwrapped
                raw._placer = DiagnosticPlacer(raw._placer, raw.backend.pan_axis_xy, raw.backend.rest_z, PoseCase("initial"))
                try:
                    for kind in ("workspace", "yaw"):
                        group = [(c, s) for c, s in cases(protocol) if (c.radius_m is not None) == (kind == "workspace")]
                        recent = [measure(env, policy, name, mass, c, s) for c, s in group]
                        rows.extend(recent)
                        write(output / "episodes.json", [asdict(r) for r in rows])
                        print(name, mass, kind, sum(r.task.final_success for r in recent), "/", len(recent), flush=True)
                finally:
                    env.close()
    finally:
        learned.close()
    return rows


def summarize(rows: list[CoverageOutcome]) -> list[dict]:
    """Keep supported-workspace and unseen-yaw results distinct for interpretation."""
    groups = []
    for name in ("scripted", "sac"):
        for kind in ("workspace", "yaw"):
            recent = [r for r in rows if r.controller == name and (r.case.radius_m is not None) == (kind == "workspace")]
            groups.append({"controller": name, "benchmark": kind, "episodes": len(recent),
                "successes": sum(r.task.final_success for r in recent),
                "not_released": sum(not r.lift_released for r in recent),
                "acquisition_aborts": sum(r.acquisition_abort is not None for r in recent),
                "table_contact_episodes": sum(r.task.robot_table_contact_steps > 0 for r in recent),
                "approach_contact_episodes": sum(r.task.approach_contact_steps > 0 for r in recent),
                "warning_episodes": sum(r.task.warnings for r in recent)})
    return groups


def run(repo: CartesianZeroRepository, output: Path) -> None:
    """Declare coverage questions before probing; never select new controller settings."""
    output.mkdir(parents=True, exist_ok=False)
    protocol, hashes = CoverageProtocol(), fingerprints(repo)
    source = Path(__file__)
    write(output / "protocol.json", {"protocol": asdict(protocol), "preset": asdict(repo.load()),
        "input_sha256": hashes, "evaluation_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "questions": ["Does the existing script cover its supported placement bounds?",
                      "Does decoupling cube yaw expose an acquisition weakness?"],
        "scope": "Workspace grid in current bounds; yaw offsets are distribution shifts, not a revised training task.",
        "decision": "Any observed weakness requires independent confirmation and authority/observability review before training."})
    rows = sweep(repo, output, protocol)
    write(output / "summary.json", summarize(rows))
    assert hashes == fingerprints(repo), "A frozen input changed"
    write(output / "input_verification.json", {"unchanged": True, "sha256": hashes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(CartesianZeroRepository(args.model_dir), args.out)

"""Plan steps 14-15 and 18: the baselines' bar, and the acceptance verdict, in the calibrated sim with stiction.

Conditions come from the preset's env options (default `residual_ik_sim2real`):
the deployment condition (every random draw fixed at the measured arm, the
longest delay, encoder rounding, LeRobot's clip) and the stress conditions
of `AcceptanceConfig` (each kp extreme, the largest stiction band). IK hold,
gravity compensation, and integral control run in delta mode at the preset's
command rate. With `--model-dir`, the trained policy runs too (capped at the
hardware command rate) and the acceptance criteria are applied; without it,
only the baselines' bar is written.

Writes to experiments/<date>_acceptance/ (or --out): README.md, report.json.

Examples:
  python scripts/sim2real/analyze/acceptance.py                       # bar to beat
  python scripts/sim2real/analyze/acceptance.py --model-dir checkpoints/reach_1m_residual_ik
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rbe_501_so101.config import PRESETS, AcceptanceConfig, HardwareConfig, RobustnessConfig, TrainConfig, resolve_project_path
from rbe_501_so101.provenance import begin_run, experiment_dir
from rbe_501_so101.rl.acceptance import AcceptanceVerdict, acceptance_conditions, check_acceptance
from rbe_501_so101.rl.checkpoints import CheckpointRepository
from rbe_501_so101.rl.policies import BASELINE_NAMES
from rbe_501_so101.rl.robustness import (
    TRAINED,
    SweepResult,
    evaluate_baselines,
    evaluate_condition,
    hardware_speed_scale,
)

DEFAULT_PRESET = "residual_ik_sim2real"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command line."""
    cfg = AcceptanceConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET, help="defines the conditions")
    parser.add_argument("--model-dir", type=Path, default=None, help="trained checkpoints to judge (optional)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TrainConfig().seeds))
    parser.add_argument("--episodes", type=int, default=cfg.episodes_per_condition, help="episodes per condition")
    parser.add_argument("--out", type=Path, default=None, help=f"default experiments/<date>_{cfg.experiment_suffix}")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, cfg: AcceptanceConfig) -> list[SweepResult]:
    """Every controller under every acceptance condition."""
    robustness = RobustnessConfig(episodes_per_point=args.episodes)
    preset = PRESETS[args.preset]
    if args.model_dir is None:
        conditions = acceptance_conditions(preset.env, cfg)
        return [r for c in conditions for r in evaluate_baselines(preset, c, robustness)]
    repo = CheckpointRepository(resolve_project_path(args.model_dir), TrainConfig())
    seeds = tuple(args.seeds)
    speed = hardware_speed_scale(repo.load_preset(seeds[0]).task, HardwareConfig())
    conditions = acceptance_conditions(preset.env, cfg, speed)
    return [r for c in conditions for r in evaluate_condition(repo, seeds, c, robustness)]


def table(results: list[SweepResult]) -> list[str]:
    """Mean (p95) holding error per condition and controller, in mm."""
    controllers = [c for c in (TRAINED, *BASELINE_NAMES) if any(r.controller == c for r in results)]
    conditions = list(dict.fromkeys((r.parameter, r.value) for r in results))
    lines = ["| condition | " + " | ".join(controllers) + " |", "|---" * (len(controllers) + 1) + "|"]
    for parameter, value in conditions:
        cells = [_cell(results, c, parameter, value) for c in controllers]
        lines.append(f"| {parameter} {value:g} | " + " | ".join(cells) + " |")
    return lines


def _cell(results: list[SweepResult], controller: str, parameter: str, value: float) -> str:
    """'mean (p95)' for one controller and condition, plus command step and (policy) saturation."""
    r = next(r for r in results if (r.controller, r.parameter, r.value) == (controller, parameter, value))
    step = "" if r.command_step_deg is None else f", {r.command_step_deg:.2f}°/step"
    sat = "" if r.saturation_fraction is None else f", sat {r.saturation_fraction:.0%}"
    return f"{r.mean_mm:.2f} ({r.p95_mm:.2f}{step}{sat})"


def readme(args: argparse.Namespace, results: list[SweepResult], verdict: AcceptanceVerdict | None) -> str:
    """Verdict (if a policy was judged) and the table."""
    head = [f"# Acceptance in the calibrated sim (`{args.preset}` conditions)", ""]
    if verdict is None:
        head += ["Baselines only: this is the bar a retrained policy must clear.", ""]
    else:
        head += [f"Checkpoint `{args.model_dir}`, seeds {args.seeds}.", "",
                 f"## Verdict: {'ACCEPT' if verdict.passed else 'REJECT'}", "", *[f"- {r}" for r in verdict.reasons], ""]
    note = ("Holding-window error, mean (p95) in mm, over the same evaluation targets; °/step = mean command change "
            "per holding step; sat = policy saturation.")
    return "\n".join([*head, note, "", *table(results)]) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args, cfg = parse_args(argv), AcceptanceConfig()
    out = resolve_project_path(args.out) if args.out else experiment_dir(cfg.experiment_suffix)
    manifest = begin_run(out)
    results = run(args, cfg)
    verdict = check_acceptance(results, cfg) if args.model_dir else None
    (out / "README.md").write_text(readme(args, results, verdict))
    report = {"preset": args.preset, "model_dir": str(args.model_dir) if args.model_dir else None, "criteria": asdict(cfg),
              "verdict": asdict(verdict) if verdict else None, "results": [asdict(r) for r in results]}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    manifest.finalize(out)
    print((out / "README.md").read_text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

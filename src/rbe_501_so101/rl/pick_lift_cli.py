"""Pick-and-lift command line: train with a fixed or random object mass, or compare runs across masses."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rbe_501_so101.config import CHECKPOINTS_DIR, ObjectMassConfig, PickLiftEvalConfig, TrainConfig
from rbe_501_so101.rl.pick_lift import MassResult, RunInfo, compare_runs, train_pick_lift_seed
from rbe_501_so101.rl.training import ALGORITHMS, resolve_device

PERCENT = 100.0
M_TO_CM = 100.0
COLUMN_WIDTH = 20


@dataclass(frozen=True)
class TrainSettings:
    """Parsed options for `train`."""

    run: RunInfo
    timesteps: int
    seeds: tuple[int, ...]
    model_dir: Path
    cpu: bool


@dataclass(frozen=True)
class CompareSettings:
    """Parsed options for `compare`."""

    model_dirs: tuple[Path, ...]
    eval_cfg: PickLiftEvalConfig
    out: Path | None
    plot: Path | None


def parse_args(cfg: TrainConfig, argv: list[str] | None = None) -> TrainSettings | CompareSettings:
    """Parse `train` or `compare` options into typed settings."""
    args = _parser(cfg).parse_args(argv)
    if args.command == "compare":
        return _compare_settings(args)
    return _train_settings(args)


def main(argv: list[str] | None = None) -> None:
    """Train every requested seed, or print (and optionally save) a mass-sweep comparison."""
    cfg = TrainConfig()
    match parse_args(cfg, argv):
        case TrainSettings() as settings:
            _train(settings, cfg)
        case CompareSettings() as settings:
            _compare(settings, cfg)


def _parser(cfg: TrainConfig) -> argparse.ArgumentParser:
    """Top-level parser with `train` and `compare` subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    _add_train_args(commands.add_parser("train", help="train SAC/PPO with a mass setting"), cfg)
    _add_compare_args(commands.add_parser("compare", help="evaluate runs across object masses"))
    return parser


def _add_train_args(parser: argparse.ArgumentParser, cfg: TrainConfig) -> None:
    """Options of the `train` subcommand."""
    mass = ObjectMassConfig()
    parser.add_argument("--mass-mode", choices=["fixed", "random"], default=mass.mode)
    parser.add_argument("--mass-kg", type=float, default=mass.fixed_kg, help="mass used by --mass-mode fixed")
    parser.add_argument("--mass-range", type=float, nargs=2, default=mass.range_kg, metavar=("LOW", "HIGH"))
    parser.add_argument("--control", choices=["residual", "direct"], default="residual")
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default=cfg.algo)
    parser.add_argument("--timesteps", type=int, default=cfg.timesteps)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.seeds))
    parser.add_argument("--model-dir", type=Path, default=None, help="default: checkpoints/pick_lift_<mode>_<mass>g_<side>mm")
    parser.add_argument("--cpu", action="store_true", help="train on CPU even if a GPU is present")


def _add_compare_args(parser: argparse.ArgumentParser) -> None:
    """Options of the `compare` subcommand."""
    eval_cfg = PickLiftEvalConfig()
    parser.add_argument("model_dirs", type=Path, nargs="*", help="run folders to compare with the scripted baseline")
    parser.add_argument("--masses", type=float, nargs="+", default=list(eval_cfg.masses_kg))
    parser.add_argument("--episodes-per-mass", type=int, default=eval_cfg.episodes_per_mass)
    parser.add_argument("--out", type=Path, default=None, help="write the comparison as JSON")
    parser.add_argument("--plot", type=Path, default=None, help="write a success/lift-vs-mass figure")


def _train_settings(args: argparse.Namespace) -> TrainSettings:
    """Typed `train` settings with the mass configuration built from the flags."""
    mass = ObjectMassConfig(mode=args.mass_mode, fixed_kg=args.mass_kg, range_kg=tuple(args.mass_range))
    run = RunInfo(mass=mass, control=args.control, algo=args.algo)
    model_dir = args.model_dir or CHECKPOINTS_DIR / _mass_run_name(run)
    return TrainSettings(run=run, timesteps=args.timesteps, seeds=tuple(args.seeds), model_dir=model_dir, cpu=args.cpu)


def _mass_run_name(run: RunInfo) -> str:
    """Name a run by mass and cube side so geometry changes use new checkpoints."""
    side = f"{run.object.side_m * 1000:g}".replace(".", "p")
    mass = run.mass
    if mass.mode == "fixed":
        return f"pick_lift_fixed_{_grams_label(mass.fixed_kg)}g_{side}mm"
    low, high = mass.range_kg
    return f"pick_lift_random_{_grams_label(low)}_{_grams_label(high)}g_{side}mm"


def _grams_label(kg: float) -> str:
    """Convert kilograms to a filesystem friendly number of grams."""
    return f"{kg * 1000:g}".replace(".", "p")


def _compare_settings(args: argparse.Namespace) -> CompareSettings:
    """Typed `compare` settings."""
    eval_cfg = PickLiftEvalConfig(masses_kg=tuple(args.masses), episodes_per_mass=args.episodes_per_mass)
    return CompareSettings(model_dirs=tuple(args.model_dirs), eval_cfg=eval_cfg, out=args.out, plot=args.plot)


def _train(settings: TrainSettings, cfg: TrainConfig) -> None:
    """Train each seed in turn."""
    device = resolve_device(prefer_gpu=not settings.cpu)
    for seed in settings.seeds:
        path = train_pick_lift_seed(seed, settings.run, settings.timesteps, settings.model_dir, cfg, device)
        print(f"saved {path}")


def _compare(settings: CompareSettings, cfg: TrainConfig) -> None:
    """Evaluate, print a table, and write the optional JSON and figure."""
    report = compare_runs(list(settings.model_dirs), settings.eval_cfg, cfg)
    print(format_table(report))
    if settings.out:
        settings.out.write_text(json.dumps({k: [asdict(r) for r in v] for k, v in report.items()}, indent=2))
    if settings.plot:
        from rbe_501_so101.rl.pick_lift_plots import plot_mass_sweep

        plot_mass_sweep(report, settings.plot)


def format_table(report: dict[str, list[MassResult]]) -> str:
    """Rows of mass, columns of policies; each cell is success rate / mean final lift."""
    names = list(report)
    header = "mass (kg)".ljust(COLUMN_WIDTH) + "".join(n.ljust(COLUMN_WIDTH) for n in names)
    rows = [header]
    for i, mass in enumerate(r.mass_kg for r in report[names[0]]):
        cells = [_cell(report[name][i]) for name in names]
        rows.append(f"{mass:<{COLUMN_WIDTH}.2f}" + "".join(c.ljust(COLUMN_WIDTH) for c in cells))
    return "\n".join(rows)


def _cell(result: MassResult) -> str:
    """'90% / 5.8 cm' style summary of one result."""
    return f"{PERCENT * result.success_rate:3.0f}% / {M_TO_CM * result.mean_lift_m:4.1f} cm"

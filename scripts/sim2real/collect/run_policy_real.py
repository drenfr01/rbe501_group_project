"""Phase 3: run one controller on the real SO-101 over a fixed target list.

Controllers: ik_hold, gravity_comp, integral (classical baselines in delta
mode) or trained (a checkpoint, VecNormalize statistics frozen). Targets are
sim world points (robot_mount assumed where the MJCF puts it). Every episode
is logged per step (states, actions, motor status, encoder-based error).

  fixed set:      --targets data/targets/reach_fixed.json   --repeats 1
  repeatability:  --targets data/targets/reach_repeat.json  --repeats 10

Press `q` between steps for a software abort (the arm holds, then ramps to
rest). The physical power switch is the e-stop; keep it within reach.
`--sim` runs the same stack on MuJoCo through the sim bus (with
`--calibration` applied to the sim arm too) as a dry run / prediction.

Writes experiments/<date>_real_reach/<controller>_<robot_id>_<HHMMSS>/.
"""

import argparse
import contextlib
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from rbe_501_so101.config import (
    PRESETS,
    ExperimentPreset,
    HardwareConfig,
    RealTrialConfig,
    SimConfig,
    TrainConfig,
    resolve_project_path,
)
from rbe_501_so101.envs.calibration import ActuatorCalibration
from rbe_501_so101.envs.reach_env import SO101ReachEnv, task_home_deg
from rbe_501_so101.envs.targets import SequenceTargetSampler
from rbe_501_so101.envs.types import ObservationLayout
from rbe_501_so101.hardware.operator import KeySource, OperatorAbortPolicy, ScriptedKeys, TerminalKeys
from rbe_501_so101.hardware.safety import SafetyAbort
from rbe_501_so101.hardware.session import build_real_backend, open_bus, robot_id_for
from rbe_501_so101.hardware.trials import (
    EpisodeRecord,
    ScheduledTrial,
    TrialRepository,
    load_targets,
    run_logged_episode,
    summarize_trials,
    trial_readme,
    trial_schedule,
)
from rbe_501_so101.provenance import begin_run, experiment_dir, require_clean_tree
from rbe_501_so101.rl.checkpoints import CheckpointRepository
from rbe_501_so101.rl.evaluation import load_trained_policy
from rbe_501_so101.rl.policies import BASELINE_NAMES, EvalPolicy, make_baseline
from rbe_501_so101.rl.robustness import TRAINED, baseline_task, slowed_preset

DEFAULT_PRESET = "residual_ik"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command line."""
    hw, trials = HardwareConfig(), RealTrialConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=(*BASELINE_NAMES, TRAINED), required=True)
    parser.add_argument("--model-dir", type=Path, default=None, help="checkpoint folder (required for trained)")
    parser.add_argument("--seed", type=int, default=0, help="checkpoint seed")
    parser.add_argument("--calibration", type=Path, default=None, help="ActuatorCalibration JSON")
    parser.add_argument("--targets", type=Path, default=trials.fixed_targets_path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--speed-scale", type=float, default=hw.speed_scale, help="scales the command rate limit")
    parser.add_argument("--port", default=hw.port)
    parser.add_argument("--robot-id", default=hw.robot_id)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sim", action="store_true", help="dry run on the MuJoCo arm")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def load_preset(args: argparse.Namespace) -> ExperimentPreset:
    """Checkpoint preset (trained) or the default preset, at the requested speed; delta mode for baselines."""
    if args.policy == TRAINED:
        preset = CheckpointRepository(resolve_project_path(args.model_dir), TrainConfig()).load_preset(args.seed)
    else:
        preset = PRESETS[DEFAULT_PRESET]
    preset = slowed_preset(preset, args.speed_scale)
    return preset if args.policy == TRAINED else replace(preset, task=baseline_task(preset.task))


def build_policy(args: argparse.Namespace, preset: ExperimentPreset, env: SO101ReachEnv) -> EvalPolicy:
    """The selected controller, with stiffness from the calibration file when given."""
    if args.policy == TRAINED:
        repo = CheckpointRepository(resolve_project_path(args.model_dir), TrainConfig())
        return load_trained_policy(repo, args.seed, TrainConfig().algo)
    calibration = ActuatorCalibration.load(resolve_project_path(args.calibration)) if args.calibration else None
    layout = ObservationLayout(n_arm=env.layout.n_arm)
    return make_baseline(args.policy, SimConfig(), preset.task, layout, calibration)


def run_trials(
    env: SO101ReachEnv, policy: EvalPolicy, schedule: tuple[ScheduledTrial, ...], repo: TrialRepository, name: str
) -> tuple[list[EpisodeRecord], str | None]:
    """Run the schedule, saving each episode; stop at the first safety abort."""
    records: list[EpisodeRecord] = []
    for k, trial in enumerate(schedule):
        try:
            record = run_logged_episode(env, policy, name, trial.target_index, trial.repeat, seed=k)
        except SafetyAbort as abort:
            return records, str(abort)
        repo.save_episode(record)
        records.append(record)
        print(f"target {trial.target_index} repeat {trial.repeat}: final error {1000 * record.steps[-1].error_m:.2f} mm")
    return records, None


def output_dir(args: argparse.Namespace, robot_id: str) -> Path:
    """Explicit `--out`, or a timestamped folder under the dated experiment."""
    if args.out:
        return resolve_project_path(args.out)
    stamp = datetime.now().strftime("%H%M%S")
    return experiment_dir(RealTrialConfig().experiment_suffix) / f"{args.policy}_{robot_id}_{stamp}"


def key_source(sim: bool) -> contextlib.AbstractContextManager[KeySource]:
    """Terminal keys on hardware with a TTY; no keys in sim or without a terminal."""
    if sim or not sys.stdin.isatty():
        return contextlib.nullcontext(ScriptedKeys())
    return TerminalKeys()


def session(args: argparse.Namespace, hw: HardwareConfig, out: Path) -> tuple[list[EpisodeRecord], str | None]:
    """Build the real stack, run the schedule under the operator abort, and always close (ramp to rest)."""
    preset = load_preset(args)
    sim_cfg = SimConfig(calibration_path=resolve_project_path(args.calibration) if args.calibration else None)
    bus, clock = open_bus(args.sim, hw, sim_cfg)
    backend = build_real_backend(bus, clock, hw, home_deg=task_home_deg(SimConfig(), preset.task))
    schedule = trial_schedule(load_targets(resolve_project_path(args.targets)), args.repeats)
    sampler = SequenceTargetSampler(tuple(t.target for t in schedule))
    env = SO101ReachEnv(backend=backend, sampler=sampler, task_cfg=preset.task, reward_cfg=preset.reward)
    try:
        policy = build_policy(args, preset, env)
        with key_source(args.sim) as keys:
            return run_trials(env, OperatorAbortPolicy(policy, keys), schedule, TrialRepository(out), args.policy)
    finally:
        env.close()


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    if args.policy == TRAINED and args.model_dir is None:
        sys.exit("--policy trained needs --model-dir")
    require_clean_tree(args.allow_dirty)
    hw = replace(HardwareConfig(), port=args.port, robot_id=args.robot_id)
    robot_id = robot_id_for(args.sim, hw)
    out = output_dir(args, robot_id)
    manifest = begin_run(out, robot_id)
    records, aborted = session(args, hw, out)
    if records:
        summary = summarize_trials(records, TrainConfig().holding_window_steps)
        TrialRepository(out).save_summary(summary)
        context = f"Robot `{robot_id}`, speed scale {args.speed_scale}, calibration `{args.calibration}`."
        (out / "README.md").write_text(trial_readme(summary, aborted, context))
        print(f"mean {summary.mean_error_mm:.2f} mm, p95 {summary.p95_error_mm:.2f} mm -> {out}")
    if aborted:
        print(f"aborted: {aborted}")
    manifest.finalize(out)


if __name__ == "__main__":
    main()

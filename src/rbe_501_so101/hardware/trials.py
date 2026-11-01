"""Phase 3 trials: per-step episode logs, their repository, and error/repeatability summaries."""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rbe_501_so101.envs.reach_env import SO101ReachEnv
from rbe_501_so101.hardware.types import MotorStatus
from rbe_501_so101.rl.policies import EvalPolicy

M_TO_MM = 1000.0
P95 = 95


@dataclass(frozen=True)
class StepRecord:
    """One control step: measured and commanded angles (rad), action, encoder-based TCP and error."""

    step: int
    qpos: tuple[float, ...]
    q_cmd: tuple[float, ...]
    action: tuple[float, ...]
    tcp_pos: tuple[float, ...]
    error_m: float


@dataclass(frozen=True)
class StatusSample:
    """Motor status read by the backend at one step."""

    step: int
    motors: tuple[MotorStatus, ...]


@dataclass(frozen=True)
class EpisodeRecord:
    """Everything logged for one hardware episode."""

    controller: str
    target_index: int
    repeat: int
    target: tuple[float, ...]
    steps: tuple[StepRecord, ...]
    status: tuple[StatusSample, ...]
    overruns: int

    def final_tcp(self) -> np.ndarray:
        """Encoder-based gripper position at the last step."""
        return np.asarray(self.steps[-1].tcp_pos)

    def holding_errors_m(self, window: int) -> np.ndarray:
        """Errors over the last `window` steps."""
        return np.array([s.error_m for s in self.steps[-window:]])


@dataclass(frozen=True)
class RepeatabilityStats:
    """Spread of repeated final positions about their centroid (mm)."""

    n: int
    mean_radius_mm: float
    max_radius_mm: float


@dataclass(frozen=True)
class TargetSummary:
    """Holding error and repeatability at one target."""

    target_index: int
    mean_error_mm: float
    p95_error_mm: float
    repeatability: RepeatabilityStats


@dataclass(frozen=True)
class TrialSummary:
    """All episodes of one controller session."""

    controller: str
    n_episodes: int
    mean_error_mm: float
    p95_error_mm: float
    total_overruns: int
    per_target: dict[int, TargetSummary]


class TrialRepository:
    """Owns one Phase 3 output folder: `episodes/` plus `summary.json`."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def save_episode(self, record: EpisodeRecord) -> Path:
        """Write one episode as JSON and return its path."""
        path = self.folder / "episodes" / f"t{record.target_index:02d}_r{record.repeat:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(record)))
        return path

    def load_episode(self, path: Path) -> EpisodeRecord:
        """Read an episode written by `save_episode`."""
        return _episode_from_dict(json.loads(path.read_text()))

    def save_summary(self, summary: TrialSummary) -> Path:
        """Write `summary.json`."""
        path = self.folder / "summary.json"
        path.write_text(json.dumps(asdict(summary), indent=2))
        return path


@dataclass(frozen=True)
class ScheduledTrial:
    """One episode in a hardware session: which target, which repeat."""

    target_index: int
    repeat: int
    target: np.ndarray


def load_targets(path: Path) -> tuple[np.ndarray, ...]:
    """Targets file: `{"frame": "world", "targets": [[x, y, z], ...]}` in sim world metres."""
    data = json.loads(Path(path).read_text())
    return tuple(np.asarray(t, dtype=float) for t in data["targets"])


def trial_schedule(targets: tuple[np.ndarray, ...], repeats: int) -> tuple[ScheduledTrial, ...]:
    """Each target `repeats` times in a row, in file order."""
    return tuple(ScheduledTrial(i, r, t) for i, t in enumerate(targets) for r in range(repeats))


def run_logged_episode(
    env: SO101ReachEnv, policy: EvalPolicy, controller: str, target_index: int, repeat: int, seed: int
) -> EpisodeRecord:
    """Roll out one episode, logging every step and any status the backend read."""
    obs, _ = env.reset(seed=seed)
    policy.reset(env.target, env.state)
    status_start = len(getattr(env.backend, "status_log", []))
    steps, truncated = [], False
    while not truncated:
        action = np.asarray(policy(obs), dtype=float)
        obs, _, _, truncated, info = env.step(action)
        steps.append(_step_record(len(steps), env, action, info["distance_m"]))
    status = tuple(StatusSample(i, m) for i, m in getattr(env.backend, "status_log", [])[status_start:])
    overruns = int(getattr(env.backend, "overrun_count", 0))
    target = tuple(float(v) for v in env.target)
    return EpisodeRecord(controller, target_index, repeat, target, tuple(steps), status, overruns)


def repeatability(final_positions: np.ndarray) -> RepeatabilityStats:
    """Distance of each final position from the centroid."""
    finals = np.asarray(final_positions, dtype=float)
    radii = np.linalg.norm(finals - finals.mean(axis=0), axis=1)
    return RepeatabilityStats(len(finals), M_TO_MM * float(radii.mean()), M_TO_MM * float(radii.max()))


def summarize_trials(records: list[EpisodeRecord], holding_window: int) -> TrialSummary:
    """Encoder-based holding error overall and per target, plus repeatability per target."""
    by_target: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_target[record.target_index].append(record)
    errors = np.concatenate([r.holding_errors_m(holding_window) for r in records])
    return TrialSummary(
        controller=records[0].controller,
        n_episodes=len(records),
        mean_error_mm=M_TO_MM * float(errors.mean()),
        p95_error_mm=M_TO_MM * float(np.percentile(errors, P95)),
        total_overruns=max(r.overruns for r in records),
        per_target={i: _target_summary(i, eps, holding_window) for i, eps in sorted(by_target.items())},
    )


def _target_summary(index: int, records: list[EpisodeRecord], window: int) -> TargetSummary:
    """Error and spread at one target."""
    errors = np.concatenate([r.holding_errors_m(window) for r in records])
    spread = repeatability(np.array([r.final_tcp() for r in records]))
    return TargetSummary(index, M_TO_MM * float(errors.mean()), M_TO_MM * float(np.percentile(errors, P95)), spread)


def _step_record(index: int, env: SO101ReachEnv, action: np.ndarray, error_m: float) -> StepRecord:
    """Snapshot the env state after a step."""
    state = env.state
    return StepRecord(
        step=index,
        qpos=tuple(state.qpos.tolist()),
        q_cmd=tuple(state.q_cmd.tolist()),
        action=tuple(action.tolist()),
        tcp_pos=tuple(state.tcp_pos.tolist()),
        error_m=float(error_m),
    )


def _episode_from_dict(data: dict[str, Any]) -> EpisodeRecord:
    """Rebuild an episode from JSON."""
    steps = tuple(StepRecord(**{k: tuple(v) if isinstance(v, list) else v for k, v in s.items()}) for s in data["steps"])
    status = tuple(StatusSample(s["step"], tuple(MotorStatus(**m) for m in s["motors"])) for s in data["status"])
    return EpisodeRecord(
        controller=data["controller"],
        target_index=data["target_index"],
        repeat=data["repeat"],
        target=tuple(data["target"]),
        steps=steps,
        status=status,
        overruns=data["overruns"],
    )


def trial_readme(summary: TrialSummary, aborted: str | None = None, context: str = "") -> str:
    """Markdown for one controller session: encoder-based error and repeatability per target."""
    lines = [
        f"# Real reach: {summary.controller}",
        "",
        *([context, ""] if context else []),
        f"{summary.n_episodes} episodes. Encoder-based holding error (forward kinematics of measured angles):",
        f"mean {summary.mean_error_mm:.2f} mm, p95 {summary.p95_error_mm:.2f} mm. Control-loop overruns: {summary.total_overruns}.",
        "",
        "| target | mean (mm) | p95 (mm) | repeats | spread mean (mm) | spread max (mm) |",
        "|---|---|---|---|---|---|",
        *(_target_row(t) for t in summary.per_target.values()),
    ]
    if aborted:
        lines += ["", f"**Session aborted:** {aborted}"]
    return "\n".join(lines) + "\n"


def _target_row(t: TargetSummary) -> str:
    """One per-target table row."""
    r = t.repeatability
    return f"| {t.target_index} | {t.mean_error_mm:.2f} | {t.p95_error_mm:.2f} | {r.n} | {r.mean_radius_mm:.2f} | {r.max_radius_mm:.2f} |"

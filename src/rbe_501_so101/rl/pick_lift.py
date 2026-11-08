"""Pick-and-lift training and mass-sweep evaluation."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol, Self

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rbe_501_so101.config import ObjectConfig, ObjectMassConfig, PickLiftConfig, PickLiftEvalConfig, ResidualConfig, TrainConfig
from rbe_501_so101.control import ScriptedPickLift
from rbe_501_so101.envs import SO101PickLiftEnv
from rbe_501_so101.envs.mujoco_backend import RenderMode
from rbe_501_so101.rl.checkpoint import TrainingCheckpoint
from rbe_501_so101.rl.residual import ResidualPickLift
from rbe_501_so101.rl.training import ALGORITHMS, AlgoName, train_with_factory

ControlMode = Literal["residual", "direct"]
RUN_INFO_FILENAME = "run_info.json"
LEGACY_CUBE_SIDE_M = 0.03
MONITOR_KEYWORDS = ("is_success", "mass_kg", "lift_m")
SCRIPTED_NAME = "scripted"

__all__ = [
    "EpisodePolicy",
    "MassResult",
    "NormalizedPolicy",
    "ResidualPickLift",
    "RunInfo",
    "compare_runs",
    "evaluate_over_masses",
    "evaluate_run",
    "load_policy",
    "load_run",
    "make_pick_lift_env",
    "pick_lift_factory",
    "train_pick_lift_seed",
]


@dataclass(frozen=True)
class RunInfo:
    """What a training run used; saved as `run_info.json` beside its seed folders."""

    mass: ObjectMassConfig
    control: ControlMode
    algo: AlgoName
    object: ObjectConfig = field(default_factory=ObjectConfig)

    def save(self, model_dir: Path) -> None:
        """Write this record into `model_dir`."""
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / RUN_INFO_FILENAME).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, model_dir: Path) -> Self:
        """Read the record written by `save`."""
        raw = json.loads((model_dir / RUN_INFO_FILENAME).read_text())
        mass = raw["mass"]
        mass_cfg = ObjectMassConfig(mode=mass["mode"], fixed_kg=mass["fixed_kg"], range_kg=tuple(mass["range_kg"]))
        object_cfg = ObjectConfig(**raw["object"]) if "object" in raw else ObjectConfig(side_m=LEGACY_CUBE_SIDE_M)
        return cls(mass=mass_cfg, control=raw["control"], algo=raw["algo"], object=object_cfg)


@dataclass(frozen=True)
class MassResult:
    """Success rate and mean final lift height over the episodes run at one mass."""

    mass_kg: float
    success_rate: float
    mean_lift_m: float
    episodes: int


class EpisodePolicy(Protocol):
    """A policy evaluated on raw observations, reset at the start of each episode."""

    def reset(self) -> None:
        """Prepare for a new episode."""
        ...

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return an action in [-1, 1]."""
        ...


class NormalizedPolicy:
    """Deterministic SB3 model behind frozen VecNormalize observation statistics."""

    def __init__(self, model: BaseAlgorithm, normalizer: VecNormalize) -> None:
        self._model = model
        self._normalizer = normalizer

    def reset(self) -> None:
        """Stateless policy; nothing to reset."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action for the normalized observation."""
        action, _ = self._model.predict(self._normalizer.normalize_obs(obs), deterministic=True)
        return action

    def close(self) -> None:
        """Release the dummy env used only to load normalization statistics."""
        self._normalizer.close()


def load_run(model_dir: Path) -> RunInfo:
    """Training record of the run stored in `model_dir`."""
    return RunInfo.load(model_dir)


def make_pick_lift_env(
    mass_cfg: ObjectMassConfig,
    control: ControlMode,
    render_mode: RenderMode | None = None,
    residual_cfg: ResidualConfig = ResidualConfig(),
    object_cfg: ObjectConfig = ObjectConfig(),
) -> gym.Env:
    """Pick-and-lift env, wrapped for residual control on top of the scripted controller if asked."""
    env = SO101PickLiftEnv(render_mode=render_mode, mass_cfg=mass_cfg, cfg=replace(PickLiftConfig(), object=object_cfg))
    if control == "residual":
        return ResidualPickLift(env, ScriptedPickLift.for_env(env), residual_cfg)
    return env


def pick_lift_factory(
    mass_cfg: ObjectMassConfig, control: ControlMode, monitor_path: Path | None = None, append_monitor: bool = False,
    object_cfg: ObjectConfig = ObjectConfig(),
) -> Callable[[], gym.Env]:
    """Create a monitored environment, optionally appending to its log."""

    def factory() -> gym.Env:
        env = make_pick_lift_env(mass_cfg, control, object_cfg=object_cfg)
        filename = str(monitor_path) if monitor_path else None
        return Monitor(env, filename=filename, info_keywords=MONITOR_KEYWORDS, override_existing=not append_monitor)

    return factory


def train_pick_lift_seed(
    seed: int, run: RunInfo, timesteps: int, model_dir: Path, cfg: TrainConfig, device: str = "cuda"
) -> Path:
    """Train or resume one seed and return its model directory."""
    run_file = model_dir / RUN_INFO_FILENAME
    if run_file.exists() and RunInfo.load(model_dir) != run:
        raise ValueError(f"{model_dir} has different training settings; choose a new --model-dir")
    run.save(model_dir)
    seed_dir = model_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    resuming = TrainingCheckpoint.in_dir(seed_dir, cfg).exists()
    factory = pick_lift_factory(run.mass, run.control, seed_dir / "train", append_monitor=resuming, object_cfg=run.object)
    return train_with_factory(seed, factory, algo=run.algo, timesteps=timesteps, seed_dir=seed_dir, cfg=cfg, device=device)


def load_policy(seed_dir: Path, run: RunInfo, cfg: TrainConfig) -> NormalizedPolicy:
    """Load one seed's checkpoint with its frozen normalization statistics."""
    venv = DummyVecEnv([pick_lift_factory(run.mass, run.control, object_cfg=run.object)])
    normalizer = VecNormalize.load(str(seed_dir / cfg.vecnormalize_filename), venv)
    normalizer.training = False
    model = ALGORITHMS[run.algo].load(seed_dir / cfg.model_filename, device="cpu")
    return NormalizedPolicy(model, normalizer)


def run_episode(env: gym.Env, policy: EpisodePolicy, seed: int) -> tuple[bool, float]:
    """Roll out one episode; return (success at the last step, final lift height in m)."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    truncated, info = False, {}
    while not truncated:
        obs, _, _, truncated, info = env.step(policy(obs))
    return bool(info["is_success"]), float(info["lift_m"])


def evaluate_at_mass(
    policy_for: Callable[[gym.Env], EpisodePolicy], control: ControlMode, mass_kg: float, eval_cfg: PickLiftEvalConfig,
    object_cfg: ObjectConfig = ObjectConfig(),
) -> MassResult:
    """Run the shared evaluation placements with the object fixed at `mass_kg`."""
    env = make_pick_lift_env(ObjectMassConfig(mode="fixed", fixed_kg=mass_kg), control, object_cfg=object_cfg)
    try:
        policy = policy_for(env)
        seeds = [eval_cfg.seed_offset + i for i in range(eval_cfg.episodes_per_mass)]
        successes, lifts = zip(*(run_episode(env, policy, s) for s in seeds))
    finally:
        env.close()
    return MassResult(mass_kg, float(np.mean(successes)), float(np.mean(lifts)), len(seeds))


def evaluate_over_masses(
    policy_for: Callable[[gym.Env], EpisodePolicy], control: ControlMode, eval_cfg: PickLiftEvalConfig,
    object_cfg: ObjectConfig = ObjectConfig(),
) -> list[MassResult]:
    """One `MassResult` per evaluation mass, in the configured order."""
    return [evaluate_at_mass(policy_for, control, m, eval_cfg, object_cfg) for m in eval_cfg.masses_kg]


def evaluate_run(model_dir: Path, eval_cfg: PickLiftEvalConfig, cfg: TrainConfig) -> list[MassResult]:
    """Evaluate every `seed_*` checkpoint of a run and pool the episodes per mass."""
    run = load_run(model_dir)
    per_seed = [_evaluate_seed(seed_dir, run, eval_cfg, cfg) for seed_dir in sorted(model_dir.glob("seed_*"))]
    if not per_seed:
        raise FileNotFoundError(f"no seed_* checkpoints in {model_dir}")
    return [_pool(results) for results in zip(*per_seed)]


def compare_runs(model_dirs: list[Path], eval_cfg: PickLiftEvalConfig, cfg: TrainConfig) -> dict[str, list[MassResult]]:
    """Scripted baseline plus each trained run, keyed by name, on identical placements and masses."""
    report = {SCRIPTED_NAME: evaluate_over_masses(ScriptedPickLift.for_env, "direct", eval_cfg)}
    for model_dir in model_dirs:
        report[model_dir.name] = evaluate_run(model_dir, eval_cfg, cfg)
    return report


def _evaluate_seed(seed_dir: Path, run: RunInfo, eval_cfg: PickLiftEvalConfig, cfg: TrainConfig) -> list[MassResult]:
    """Load one checkpoint once and sweep it across all masses."""
    policy = load_policy(seed_dir, run, cfg)
    try:
        return evaluate_over_masses(lambda _env: policy, run.control, eval_cfg, run.object)
    finally:
        policy.close()


def _pool(results: tuple[MassResult, ...]) -> MassResult:
    """Episode-weighted average of several seeds' results at the same mass."""
    episodes = sum(r.episodes for r in results)
    success = sum(r.success_rate * r.episodes for r in results) / episodes
    lift = sum(r.mean_lift_m * r.episodes for r in results) / episodes
    return MassResult(results[0].mass_kg, success, lift, episodes)

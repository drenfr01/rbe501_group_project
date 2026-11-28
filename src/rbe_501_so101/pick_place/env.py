"""Pick-and-place on the SO-101, shared by both experiments: settings, env, reward, scripted controller,
training, evaluation, and the command line of `scripts/pick_place_fixed.py` and `scripts/pick_place_random.py`.

Task: pick the cube up anywhere in the workspace and set it down on a goal; the cube mass m is a training parameter.
Start and goal are random, or chosen with `env.reset(options={"object_xy": (x, y), "goal_xy": (x, y)})` in {s}.

Success: the cube rests on the table within `place_tolerance_m` of the goal, it was let go low and above the goal
(no drops, no pushing it there afterwards), and neither finger pad touches it anymore.

Built on the pick-and-lift code (env, backend, scripted grasp, residual wrapper, SAC training) without changing it.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Self

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rbe_501_so101.config import (
    CHECKPOINTS_DIR,
    GripperConfig,
    ObjectConfig,
    ObjectMassConfig,
    ObjectPlacementConfig,
    PickLiftConfig,
    PickLiftTaskConfig,
    ResidualConfig,
    ScriptedPickLiftConfig,
    TrainConfig,
)
from rbe_501_so101.control.scripted_pick_lift import UP_S, Phase, PhaseTarget, ScriptedPickLift
from rbe_501_so101.envs.grasp_model import FIXED_PAD_GEOM, MOVING_PAD_GEOM
from rbe_501_so101.envs.mujoco_backend import RenderMode
from rbe_501_so101.envs.pick_lift_env import SO101PickLiftEnv
from rbe_501_so101.envs.types import GraspState, PickLiftObservationLayout
from rbe_501_so101.kinematics.pinch_ik import PinchIKSolver
from rbe_501_so101.rl.checkpoint import TrainingCheckpoint
from rbe_501_so101.rl.pick_lift import NormalizedPolicy
from rbe_501_so101.rl.residual import ResidualPickLift
from rbe_501_so101.rl.training import ALGORITHMS, resolve_device, train_with_factory

# ================================================================ settings

CHECKPOINTS = CHECKPOINTS_DIR / "pick-place"
RUN_INFO = "run_info.json"
RESET_KEYS = ("object_xy", "goal_xy")
MONITOR_KEYWORDS = ("is_success", "mass_kg", "lift_m", "goal_distance_m")

WORKSPACE = ObjectPlacementConfig(radial_min_m=0.10, radial_max_m=0.24, azimuth_max_deg=90.0)
"""Scripted top-down grasp works at 0.10-0.25 m and +-100 deg around the pan axis (50-300 g); kept with a margin."""

EVAL_MASSES_KG = (0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
"""Training range plus two heavier masses no policy saw."""

EVAL_EPISODES = 10
EVAL_SEED_OFFSET = 20_000


@dataclass(frozen=True)
class PlaceConfig:
    """Success rules, random-goal spacing, and staged reward weights."""

    place_tolerance_m: float = 0.02
    rest_tolerance_m: float = 0.005
    max_release_height_m: float = 0.015
    carry_height_m: float = 0.04
    carry_clearance_m: float = 0.01
    min_separation_m: float = 0.06
    max_goal_tries: int = 1000
    reach_scale_per_m: float = 10.0
    goal_scale_per_m: float = 10.0
    reach_weight: float = 0.25
    grasp_weight: float = 0.35
    lift_weight: float = 0.5
    carry_weight: float = 0.75
    place_weight: float = 0.9
    success_weight: float = 1.0
    action_penalty: float = 0.01
    smoothness_penalty: float = 0.0
    goal_rgba: tuple[float, float, float, float] = (0.1, 0.8, 0.2, 0.6)


@dataclass(frozen=True)
class ScriptedPickPlaceConfig(ScriptedPickLiftConfig):
    """Scripted grasp waypoints plus carry, lower, release, and retreat."""

    lift_steps: int = 40
    carry_steps: int = 80
    lower_steps: int = 50
    release_steps: int = 20
    carry_height_m: float = 0.07
    place_height_m: float = -0.003
    carry_rate_rad: float = 0.03
    lower_ramp_steps: int = 35
    retreat_ramp_steps: int = 25


@dataclass(frozen=True)
class PickPlaceConfig(PickLiftConfig):
    """Pick-and-lift settings with the wider workspace, a longer episode, and the place rules."""

    task: PickLiftTaskConfig = field(default_factory=lambda: PickLiftTaskConfig(max_episode_steps=320))
    placement: ObjectPlacementConfig = field(default_factory=lambda: WORKSPACE)
    place: PlaceConfig = field(default_factory=PlaceConfig)


# ================================================================ reward


@dataclass(frozen=True)
class PickPlaceRewardTerms:
    """Value of every stage, their max (`staged`), the action costs, and measured distances."""

    reach: float
    grasp: float
    lift: float
    carry: float
    place: float
    success: float
    staged: float
    action_penalty: float
    smoothness_penalty: float
    total: float
    distance_m: float
    lift_m: float
    goal_distance_m: float


class PickPlaceReward:
    """Staged reward: max(reach, grasp, lift, carry, place, success) - c ||a||^2 - s ||a - a_prev||^2.

    Each stage tops out where the next starts, so the reward only grows as the task advances.
    `pads` is how many finger pads touch the cube; `p_release` is where it was let go (None while held).
    """

    def __init__(self, cfg: PlaceConfig = PlaceConfig()) -> None:
        self._cfg = cfg

    def __call__(
        self, s: GraspState, pads: int, p_goal: np.ndarray, p_release: np.ndarray | None,
        action: np.ndarray, prev: np.ndarray,
    ) -> PickPlaceRewardTerms:
        """Decomposed reward for one step."""
        c = self._cfg
        goal_d = xy_distance(s.obj.p_s, p_goal)
        distance = float(np.linalg.norm(s.obj.p_s - s.p_pinch))
        reach = c.reach_weight * (1.0 - np.tanh(c.reach_scale_per_m * distance))
        grasp, lift, carry, lower = self._held_stages(s, goal_d) if s.grasped else (0.0, 0.0, 0.0, 0.0)
        place = max(lower, c.place_weight if self.is_placed(s, p_goal, p_release) else 0.0)
        success = c.success_weight if self.is_success(s, pads, p_goal, p_release) else 0.0
        staged = float(max(reach, grasp, lift, carry, place, success))
        action, prev = np.asarray(action, dtype=float), np.asarray(prev, dtype=float)
        a_cost = c.action_penalty * float(action @ action)
        s_cost = c.smoothness_penalty * float((action - prev) @ (action - prev))
        terms = (reach, grasp, lift, carry, place, success, staged, a_cost, s_cost, staged - a_cost - s_cost)
        return PickPlaceRewardTerms(*map(float, terms), distance, s.obj.lift_m, goal_d)

    def is_placed(self, s: GraspState, p_goal: np.ndarray, p_release: np.ndarray | None) -> bool:
        """Cube resting at the goal, let go low and above the goal, and not grasped."""
        c = self._cfg
        if s.grasped or p_release is None:
            return False
        resting = abs(s.obj.lift_m) <= c.rest_tolerance_m and xy_distance(s.obj.p_s, p_goal) <= c.place_tolerance_m
        released_low = p_release[2] - s.obj.rest_z <= c.max_release_height_m
        return resting and released_low and xy_distance(p_release, p_goal) <= c.place_tolerance_m

    def is_success(self, s: GraspState, pads: int, p_goal: np.ndarray, p_release: np.ndarray | None) -> bool:
        """Placed, and no finger pad touches the cube anymore."""
        return self.is_placed(s, p_goal, p_release) and pads == 0

    def _held_stages(self, s: GraspState, goal_d: float) -> tuple[float, float, float, float]:
        """Grasp, lift, carry, and lower-at-goal values while both pads hold the cube."""
        c = self._cfg
        h = float(np.clip(s.obj.lift_m / c.carry_height_m, 0.0, 1.0))
        near_goal = 1.0 - np.tanh(c.goal_scale_per_m * goal_d)
        lift = c.grasp_weight + (c.lift_weight - c.grasp_weight) * h
        carry = c.lift_weight + (c.carry_weight - c.lift_weight) * near_goal if s.obj.lift_m >= c.carry_clearance_m else 0.0
        lower = c.carry_weight + (c.place_weight - c.carry_weight) * (1.0 - h) if goal_d <= c.place_tolerance_m else 0.0
        return c.grasp_weight, lift, carry, lower


def xy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two points in the table plane."""
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


# ================================================================ environment


@dataclass(frozen=True)
class PickPlaceObservationLayout(PickLiftObservationLayout):
    """Pick-lift observation followed by the goal p_goal and the error p_goal - p_obj."""

    @property
    def goal_pos(self) -> slice:
        """Goal (resting cube center) in {s}."""
        return slice(self.object_error.stop, self.object_error.stop + self.xyz)

    @property
    def goal_error(self) -> slice:
        """Goal minus cube center."""
        return slice(self.goal_pos.stop, self.goal_pos.stop + self.xyz)

    @property
    def size(self) -> int:
        """Total observation length."""
        return self.goal_error.stop


class SO101PickPlaceEnv(SO101PickLiftEnv):
    """Pick the cube up anywhere in the workspace and place it on a random or chosen goal."""

    def __init__(self, render_mode: RenderMode | None = None, mass_cfg: ObjectMassConfig | None = None,
                 cfg: PickPlaceConfig | None = None) -> None:
        cfg = cfg or PickPlaceConfig()
        super().__init__(render_mode, mass_cfg, cfg)
        self._layout = PickPlaceObservationLayout(len(cfg.sim.joints))
        self._init_spaces()
        self._reward = PickPlaceReward(cfg.place)
        model = self.backend.model
        self._cube_geom = model.geom(cfg.object.geom).id
        self._pad_geoms = {model.geom(FIXED_PAD_GEOM).id, model.geom(MOVING_PAD_GEOM).id}
        model.site_rgba[model.site(cfg.sim.target_site).id] = cfg.place.goal_rgba
        self.p_goal = np.zeros(3)
        self.p_release: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        """Home the arm, then place the cube and the goal (random, or from `options`)."""
        gym.Env.reset(self, seed=seed)
        self.backend.reset(self.np_random)
        p_s, yaw = self._start_pose(options or {})
        self.backend.set_target_marker(self.p_goal)
        self.backend.place_object(p_s, yaw, self._cfg.mass.sample(self.np_random))
        self._state = self.backend.grasp_state()
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._step_count, self.p_release = 0, None
        self._render_human()
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Move the arm and gripper, remember where the cube is let go, and score the step."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        was_grasped, task = self._state.grasped, self._cfg.task
        self.backend.apply_delta(action[: self._n_arm].astype(float) * task.delta_q_max_rad)
        self.backend.apply_gripper_delta(float(action[self._n_arm]) * task.delta_gripper_max_rad)
        self.backend.step()
        s = self._state = self.backend.grasp_state()
        self.p_release = None if s.grasped else (s.obj.p_s.copy() if was_grasped else self.p_release)
        terms = self._reward(s, self.pads_touching(), self.p_goal, self.p_release, action, self._prev_action)
        self._prev_action = action
        self._step_count += 1
        self._render_human()
        info = self._info() | {f"reward_{k}": v for k, v in asdict(terms).items()}
        return self._observation(), terms.total, False, self._step_count >= task.max_episode_steps, info

    def contains(self, xy: np.ndarray) -> bool:
        """True if the world XY point is inside the workspace band around the pan axis."""
        w, d = self._cfg.placement, np.asarray(xy, dtype=float)[:2] - self.backend.pan_axis_xy
        azimuth = abs(np.rad2deg(np.arctan2(d[1], d[0])))
        return w.radial_min_m <= np.hypot(*d) <= w.radial_max_m and azimuth <= w.azimuth_max_deg

    def pads_touching(self) -> int:
        """How many finger pads (0, 1, 2) touch the cube."""
        data, pads = self.backend.data, set()
        for c in data.contact[: data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if self._cube_geom in pair:
                pads |= pair & self._pad_geoms
        return len(pads)

    def _start_pose(self, options: dict[str, Any]) -> tuple[np.ndarray, float]:
        """Cube start (p_s, yaw) and goal p_goal: from the options when given, random otherwise."""
        if unknown := sorted(set(options) - set(RESET_KEYS)):
            raise ValueError(f"unknown reset option(s) {unknown}; use {list(RESET_KEYS)}")
        start, goal = options.get("object_xy"), options.get("goal_xy")
        p_s, yaw = self._placer.sample(self.np_random) if start is None else self._pose_at(start, "object_xy")
        self.p_goal = self._random_goal(p_s) if goal is None else self._pose_at(goal, "goal_xy")[0]
        return p_s, yaw

    def _pose_at(self, xy: Any, name: str) -> tuple[np.ndarray, float]:
        """Resting pose at a chosen table point, yaw facing the pan axis; raise if it is unreachable."""
        xy = np.asarray(xy, dtype=float)[:2]
        if not self.contains(xy):
            raise ValueError(f"{name}={xy.tolist()} is outside the workspace {self._cfg.placement}")
        d = xy - self.backend.pan_axis_xy
        return np.array([xy[0], xy[1], self.backend.rest_z]), float(np.arctan2(d[1], d[0]))

    def _random_goal(self, p_s: np.ndarray) -> np.ndarray:
        """Random goal in the workspace at least `min_separation_m` from the cube start p_s."""
        place = self._cfg.place
        for _ in range(place.max_goal_tries):
            p_goal, _ = self._placer.sample(self.np_random)
            if xy_distance(p_goal, p_s) >= place.min_separation_m:
                return p_goal
        raise RuntimeError(f"no goal {place.min_separation_m} m from {p_s} after {place.max_goal_tries} tries")

    def _observation(self) -> np.ndarray:
        """Pick-lift observation followed by p_goal and p_goal - p_obj."""
        p_obj = self._state.obj.p_s
        return np.concatenate((super()._observation(), self.p_goal, self.p_goal - p_obj)).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        """Mass, distances, lift, grasp, placement, and success flags."""
        s, pads = self._state, self.pads_touching()
        return {
            "mass_kg": s.obj.mass_kg,
            "distance_m": float(np.linalg.norm(s.obj.p_s - s.p_pinch)),
            "lift_m": s.obj.lift_m,
            "grasped": s.grasped,
            "goal_distance_m": xy_distance(s.obj.p_s, self.p_goal),
            "placed": self._reward.is_placed(s, self.p_goal, self.p_release),
            "is_success": self._reward.is_success(s, pads, self.p_goal, self.p_release),
        }


# ================================================================ scripted controller


class PlacePhase(StrEnum):
    """Stages after the scripted grasp and lift, in execution order."""

    CARRY = "carry"
    LOWER = "lower"
    RELEASE = "release"
    RETREAT = "retreat"


class ScriptedPickPlace(ScriptedPickLift):
    """The scripted grasp, then carry to the goal, lower, open, and back away upward.

    Waypoints are pinch-point IK goals in {s}: p_goal + h z_s. LOWER and RETREAT re-solve the IK every step for a
    height that ramps linearly, so the cube moves straight down and the open jaw straight up (rate-limiting the joints
    toward one far waypoint curves the path and drags the cube on the table).
    """

    def __init__(self, solver: PinchIKSolver, layout: PickPlaceObservationLayout, task: PickLiftTaskConfig,
                 gripper: GripperConfig, cfg: ScriptedPickPlaceConfig = ScriptedPickPlaceConfig()) -> None:
        super().__init__(solver, layout, task, gripper, cfg)
        self._place_cfg = cfg

    @classmethod
    def for_env(cls, env: gym.Env, cfg: ScriptedPickPlaceConfig = ScriptedPickPlaceConfig()) -> Self:
        """Controller sharing the env's MuJoCo model, pinch offset, and goal layout."""
        return super().for_env(env, cfg)

    def reset(self) -> None:
        """Start a new episode from the first phase."""
        super().reset()
        self._phase_start = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Normalized action; LOWER and RETREAT get a fresh straight-line target every step."""
        phase = self._phase_at(self._step)
        if phase not in (PlacePhase.LOWER, PlacePhase.RETREAT):
            return super().__call__(obs)
        if phase != self._phase:
            self._phase, self._phase_start = phase, self._step
        self._target = self._vertical_target(phase, obs, self._step - self._phase_start)
        self._step += 1
        return self._track(obs, self._target)

    def _phase_at(self, step: int) -> Phase | PlacePhase:
        """Walk the schedule of phase lengths; RETREAT once it runs out."""
        c = self._place_cfg
        schedule = [(Phase.APPROACH, c.approach_steps), (Phase.DESCEND, c.descend_steps), (Phase.CLOSE, c.close_steps),
                    (Phase.LIFT, c.lift_steps), (PlacePhase.CARRY, c.carry_steps), (PlacePhase.LOWER, c.lower_steps),
                    (PlacePhase.RELEASE, c.release_steps)]
        for phase, length in schedule:
            if step < length:
                return phase
            step -= length
        return PlacePhase.RETREAT

    def _enter(self, phase: Phase | PlacePhase, obs: np.ndarray) -> PhaseTarget:
        """Waypoint for CARRY and RELEASE; the grasp phases are inherited."""
        c = self._place_cfg
        if phase == PlacePhase.CARRY:
            p_goal = obs[self._layout.goal_pos].astype(float)
            return self._move(p_goal + c.carry_height_m * UP_S, self._q_cmd(obs), closed=True, rate=c.carry_rate_rad)
        if phase == PlacePhase.RELEASE:
            return PhaseTarget(self._target.q_arm, self._jaw(closed=False), c.arm_rate_rad, c.close_rate_rad)
        return super()._enter(phase, obs)

    def _vertical_target(self, phase: PlacePhase, obs: np.ndarray, k: int) -> PhaseTarget:
        """Target on step k of LOWER (carry height -> table, closed) or RETREAT (table -> pregrasp, open)."""
        c = self._place_cfg
        if phase == PlacePhase.LOWER:
            start, end, ramp, closed = c.carry_height_m, c.place_height_m, c.lower_ramp_steps, True
        else:
            start, end, ramp, closed = c.place_height_m, c.pregrasp_height_m, c.retreat_ramp_steps, False
        h = start + (end - start) * min(1.0, (k + 1) / ramp)
        p_goal = obs[self._layout.goal_pos].astype(float)
        return self._move(p_goal + h * UP_S, self._q_cmd(obs), closed=closed, rate=c.arm_rate_rad)

    def _move(self, goal: np.ndarray, q_cmd: np.ndarray, closed: bool, rate: float) -> PhaseTarget:
        """Top-down pinch-point move to `goal` with the jaw open or closed."""
        q = self._ik(goal, q_cmd, top_down=True)
        return PhaseTarget(q, self._jaw(closed), rate, self._place_cfg.close_rate_rad)

    def _jaw(self, closed: bool) -> float:
        """Gripper angle (rad) for a closed or open jaw."""
        return float(np.deg2rad(self._gripper.close_deg if closed else self._gripper.open_deg))


# ================================================================ training and evaluation


@dataclass(frozen=True)
class PlaceRun:
    """What a training run used; saved as `run_info.json` beside its seed folders.

    Residual policy: executed action = clip(scripted + residual_scale * a). v2 keeps the correction small
    (scale 0.2), lets the script keep the gripper (arm_only), and subtracts residual_penalty * ||a||^2 from the
    reward so the policy only corrects where it pays off. v1 (the first runs: 0.5, no penalty, gripper too)
    overrode the script and placed worse than it.
    """

    mass: ObjectMassConfig
    control: str = "residual"
    algo: str = "sac"
    object: ObjectConfig = field(default_factory=ObjectConfig)
    task: str = "place"
    residual_scale: float = 0.2
    residual_penalty: float = 0.1
    arm_only: bool = True
    tag: str = "v2"

    @property
    def name(self) -> str:
        """Folder name, e.g. pick_place_fixed_1000g_25mm or pick_place_random_50_1000g_25mm."""
        side, tag = _grams(self.object.side_m), f"_{self.tag}" if self.tag else ""
        if self.mass.mode == "fixed":
            return f"pick_place_fixed_{_grams(self.mass.fixed_kg)}g_{side}mm{tag}"
        return f"pick_place_random_{_grams(self.mass.range_kg[0])}_{_grams(self.mass.range_kg[1])}g_{side}mm{tag}"

    def save(self, model_dir: Path) -> None:
        """Write this record into `model_dir`."""
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / RUN_INFO).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, model_dir: Path) -> Self:
        """Read the record written by `save`; records from before v2 load with the v1 residual settings."""
        raw = json.loads((model_dir / RUN_INFO).read_text())
        m = raw["mass"]
        mass = ObjectMassConfig(mode=m["mode"], fixed_kg=m["fixed_kg"], range_kg=tuple(m["range_kg"]))
        return cls(
            mass, raw["control"], raw["algo"], ObjectConfig(**raw["object"]), raw.get("task", "lift"),
            residual_scale=raw.get("residual_scale", 0.5), residual_penalty=raw.get("residual_penalty", 0.0),
            arm_only=raw.get("arm_only", False), tag=raw.get("tag", ""),
        )


def _grams(kg: float) -> str:
    """Kilograms as a filesystem-friendly number of grams (meters work the same way for millimeters)."""
    return f"{kg * 1000:g}".replace(".", "p")


@dataclass(frozen=True)
class MassResult:
    """Success rate and mean placement error at one mass; `seed_rates` keeps each seed's success rate."""

    mass_kg: float
    success_rate: float
    mean_error_m: float
    episodes: int
    seed_rates: tuple[float, ...] = ()

    @property
    def seed_std(self) -> float:
        """Sample standard deviation of the success rate across seeds (0 for a single seed)."""
        return float(np.std(self.seed_rates, ddof=1)) if len(self.seed_rates) > 1 else 0.0


class PlaceResidual(ResidualPickLift):
    """Residual policy on the scripted controller, with the run's scale, penalty, and arm-only setting."""

    def __init__(self, env: SO101PickPlaceEnv, run: PlaceRun) -> None:
        super().__init__(env, ScriptedPickPlace.for_env(env), ResidualConfig(scale=run.residual_scale))
        self._penalty, self._arm_only = run.residual_penalty, run.arm_only

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step with the (masked) correction; the reward pays penalty * ||a||^2 for it."""
        a = np.asarray(action, dtype=np.float32).copy()
        if self._arm_only:
            a[-1] = 0.0
        obs, reward, terminated, truncated, info = super().step(a)
        info["residual_penalty"] = self._penalty * float(a @ a)
        return obs, reward - info["residual_penalty"], terminated, truncated, info


def make_env(mass: ObjectMassConfig, run: PlaceRun | None = None, render_mode: RenderMode | None = None) -> gym.Env:
    """Pick-and-place env: bare (for the scripted controller) if `run` is None, else wrapped for its residual policy."""
    object_cfg = run.object if run else ObjectConfig()
    env = SO101PickPlaceEnv(render_mode, mass, replace(PickPlaceConfig(), object=object_cfg))
    return PlaceResidual(env, run) if run and run.control == "residual" else env


def train(run: PlaceRun, seed: int, timesteps: int, model_dir: Path, cfg: TrainConfig = TrainConfig(),
          device: str = "cuda") -> Path:
    """Train (or resume up to `timesteps`) one seed and return its folder."""
    if (model_dir / RUN_INFO).exists() and not _same_setting(PlaceRun.load(model_dir), run):
        raise ValueError(f"{model_dir} has different training settings; choose another folder")
    run.save(model_dir)
    seed_dir = model_dir / f"seed_{seed}"
    resuming = TrainingCheckpoint.in_dir(seed_dir, cfg).exists()

    def factory() -> gym.Env:
        env = make_env(run.mass, run)
        return Monitor(env, str(seed_dir / "train"), info_keywords=MONITOR_KEYWORDS, override_existing=not resuming)

    return train_with_factory(seed, factory, algo=run.algo, timesteps=timesteps, seed_dir=seed_dir, cfg=cfg, device=device)


def _setting(r: PlaceRun) -> tuple:
    """Everything that changes what a run learns: task, mass (the folder name), controller, cube, residual setup."""
    return (r.task, r.name, r.control, r.algo, r.object, r.residual_scale, r.residual_penalty, r.arm_only)


def _same_setting(a: PlaceRun, b: PlaceRun) -> bool:
    """True if a checkpoint trained with `a` can be resumed with `b`."""
    return _setting(a) == _setting(b)


def load_policy(seed_dir: Path, run: PlaceRun, cfg: TrainConfig = TrainConfig()) -> NormalizedPolicy:
    """One trained seed with its frozen observation normalization."""
    venv = DummyVecEnv([lambda: make_env(run.mass, run)])
    normalizer = VecNormalize.load(str(seed_dir / cfg.vecnormalize_filename), venv)
    normalizer.training = False
    return NormalizedPolicy(ALGORITHMS[run.algo].load(seed_dir / cfg.model_filename, device="cpu"), normalizer)


def run_episode(env: gym.Env, policy: Callable[[np.ndarray], np.ndarray], seed: int) -> tuple[bool, float]:
    """Roll out one episode; return (success, placement error in m) at its last step."""
    obs, _ = env.reset(seed=seed)
    getattr(policy, "reset", lambda: None)()
    truncated, info = False, {}
    while not truncated:
        obs, _, _, truncated, info = env.step(policy(obs))
    return bool(info["is_success"]), float(info["goal_distance_m"])


def evaluate(policy_for: Callable[[gym.Env], Any], run: PlaceRun | None, masses: tuple[float, ...] = EVAL_MASSES_KG,
             episodes: int = EVAL_EPISODES) -> list[MassResult]:
    """Run the same start/goal seeds at every mass (run=None: the scripted controller); one result per mass."""
    results = []
    for mass in masses:
        env = make_env(ObjectMassConfig(mode="fixed", fixed_kg=mass), run)
        policy = policy_for(env)
        outcomes = [run_episode(env, policy, EVAL_SEED_OFFSET + i) for i in range(episodes)]
        env.close()
        successes, errors = zip(*outcomes)
        rate = float(np.mean(successes))
        results.append(MassResult(mass, rate, float(np.mean(errors)), episodes, (rate,)))
    return results


def compare(model_dirs: list[Path], masses: tuple[float, ...] = EVAL_MASSES_KG,
            episodes: int = EVAL_EPISODES) -> dict[str, list[MassResult]]:
    """Scripted controller plus every trained run (all seeds pooled), on identical starts, goals, and masses."""
    report = {"scripted": evaluate(ScriptedPickPlace.for_env, None, masses, episodes)}
    for model_dir in model_dirs:
        run = PlaceRun.load(model_dir)
        if run.task != "place":
            raise ValueError(f"{model_dir} is a {run.task!r} run, not pick-and-place")
        per_seed = []
        for seed_dir in sorted(d for d in model_dir.glob("seed_*") if (d / TrainConfig().model_filename).exists()):
            policy = load_policy(seed_dir, run)
            per_seed.append(evaluate(lambda _env: policy, run, masses, episodes))
            policy.close()
        report[model_dir.name] = [_pool(results) for results in zip(*per_seed)]
    return report


def trained_runs(tag: str, root: Path = CHECKPOINTS) -> list[Path]:
    """Pick-and-place run folders with at least one trained seed and the given version tag (older versions skipped)."""
    model = TrainConfig().model_filename
    dirs = [d for d in sorted(root.glob("pick_place_*")) if any(d.glob(f"seed_*/{model}"))]
    return [d for d in dirs if PlaceRun.load(d).tag == tag]


def _pool(results: tuple[MassResult, ...]) -> MassResult:
    """Average several seeds' results at the same mass (equal episodes per seed), keeping each seed's rate."""
    rates = tuple(r.success_rate for r in results)
    return MassResult(results[0].mass_kg, float(np.mean(rates)), float(np.mean([r.mean_error_m for r in results])),
                      sum(r.episodes for r in results), rates)


# ================================================================ statistics: is run A better than run B?


def episode_z_test(a: MassResult, b: MassResult) -> tuple[float, float]:
    """Two-proportion z test on pooled episodes; returns (z, two-sided p). Treats every episode as independent."""
    p_pool = (a.success_rate * a.episodes + b.success_rate * b.episodes) / (a.episodes + b.episodes)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / a.episodes + 1.0 / b.episodes))
    if se == 0.0:
        return 0.0, 1.0
    z = (a.success_rate - b.success_rate) / se
    return z, math.erfc(abs(z) / math.sqrt(2.0))


def seed_permutation_p(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Exact two-sided permutation test on per-seed success rates: share of seed splits with a gap >= observed.

    Seeds are the real repeats of an experiment, so this is the honest test; with 5 vs 5 seeds the smallest
    possible p is 2/252 = 0.008, with 3 vs 3 it is 2/20 = 0.1.
    """
    rates, n = np.array(a + b), len(a)
    observed = abs(np.mean(a) - np.mean(b))
    gaps = [abs(rates[list(i)].mean() - np.delete(rates, list(i)).mean()) for i in combinations(range(len(rates)), n)]
    return float(np.mean([g >= observed - 1e-12 for g in gaps]))


def significance(report: dict[str, list[MassResult]]) -> str:
    """Per-mass gap between the first two trained runs, with the episode z test and the seed permutation test."""
    names = [n for n in report if n != "scripted"]
    if len(names) < 2:
        return ""
    a, b = names[:2]
    rows = [f"{a} - {b}: mass (kg), gap (points), z, p (episodes), p (seeds)"]
    for ra, rb in zip(report[a], report[b]):
        z, p_ep = episode_z_test(ra, rb)
        p_seed = seed_permutation_p(ra.seed_rates, rb.seed_rates)
        rows.append(f"{ra.mass_kg:<6g} {100 * (ra.success_rate - rb.success_rate):+6.1f} {z:+6.2f} {p_ep:7.3f} {p_seed:7.3f}")
    return "\n".join(rows)


# ================================================================ command line (used by the two experiment scripts)


def run_cli(run: PlaceRun, description: str) -> None:
    """`train`, `watch`, or `compare` for one experiment, e.g. the fixed-mass or the random-mass run."""
    args = _parse_args(description)
    model_dir = CHECKPOINTS / run.name
    if args.command == "train":
        for seed in args.seeds:
            print("saved", train(run, seed, args.timesteps, model_dir, device=resolve_device(not args.cpu)))
    elif args.command == "watch":
        watch(None if args.scripted else model_dir, args)
    else:
        report = compare(trained_runs(run.tag), tuple(args.masses), args.episodes)
        print(table(report))
        print(significance(report))
        if args.plot:
            plot(report, args.plot)


def _parse_args(description: str) -> argparse.Namespace:
    """Options of the three subcommands."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train", help="train residual SAC (reruns resume from the last checkpoint)")
    fit.add_argument("--seeds", type=int, nargs="+", default=[0])
    fit.add_argument("--timesteps", type=int, default=TrainConfig().timesteps, help="total steps per seed")
    fit.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is present")
    show = sub.add_parser("watch", help="play this run (or the scripted controller) in the viewer")
    point = ("RADIUS_M", "AZIMUTH_DEG")
    show.add_argument("--scripted", action="store_true", help="the scripted controller instead of the trained policy")
    show.add_argument("--seed", type=int, default=0, help="which trained seed to load")
    show.add_argument("--start", type=float, nargs=2, metavar=point, help="cube start (0.10-0.24 m, +-90 deg)")
    show.add_argument("--goal", type=float, nargs=2, metavar=point, help="place goal (0.10-0.24 m, +-90 deg)")
    show.add_argument("--mass-kg", type=float, default=0.5)
    show.add_argument("--episodes", type=int, default=3)
    show.add_argument("--headless", action="store_true", help="no viewer, just print the results")
    cmp = sub.add_parser("compare", help="scripted vs every trained run, same starts, goals, and masses")
    cmp.add_argument("--masses", type=float, nargs="+", default=list(EVAL_MASSES_KG))
    cmp.add_argument("--episodes", type=int, default=EVAL_EPISODES, help="per mass and seed")
    cmp.add_argument("--plot", type=Path, help="save success rate and placement error vs mass")
    return parser.parse_args()


def watch(model_dir: Path | None, args: argparse.Namespace) -> None:
    """Play episodes (trained seed, or scripted if `model_dir` is None) and print each placement error."""
    mass, render = ObjectMassConfig(mode="fixed", fixed_kg=args.mass_kg), None if args.headless else "human"
    if model_dir is None:
        env = make_env(mass, None, render)
        policy = ScriptedPickPlace.for_env(env)
    else:
        run = PlaceRun.load(model_dir)
        env = make_env(mass, run, render)
        policy = load_policy(model_dir / f"seed_{args.seed}", run)
    axis = env.unwrapped.backend.pan_axis_xy
    polar = {"object_xy": args.start, "goal_xy": args.goal}
    options = {k: axis + r * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
               for k, (r, a) in ((k, p) for k, p in polar.items() if p is not None)}
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=EVAL_SEED_OFFSET + episode, options=options)
            policy.reset()
            truncated = False
            while not truncated and env.unwrapped.viewer_open:
                obs, _, _, truncated, info = env.step(policy(obs))
                time.sleep(0 if args.headless else env.unwrapped.control_dt)
            if not env.unwrapped.viewer_open:
                break
            print(f"episode {episode}: {'SUCCESS' if info['is_success'] else 'fail'}, "
                  f"placement error {100 * info['goal_distance_m']:.1f} cm")
    except ValueError as err:
        raise SystemExit(f"error: {err}") from None
    finally:
        env.close()


def table(report: dict[str, list[MassResult]]) -> str:
    """Rows of mass, columns of policies; each cell is 'success % +- spread across seeds / mean placement error'."""
    names = list(report)
    widths = [max(24, len(n) + 2) for n in names]
    rows = ["mass (kg)".ljust(10) + "".join(n.ljust(w) for n, w in zip(names, widths))]
    for i, mass in enumerate(r.mass_kg for r in report[names[0]]):
        cells = [_cell(report[n][i]) for n in names]
        rows.append(f"{mass:<10g}" + "".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(rows)


def _cell(r: MassResult) -> str:
    """'success % +-seed spread / mean error', e.g. ' 86 +- 4% / 1.9 cm'."""
    return f"{100 * r.success_rate:3.0f} +-{100 * r.seed_std:2.0f}% / {100 * r.mean_error_m:4.1f} cm"


def plot(report: dict[str, list[MassResult]], path: Path) -> None:
    """Success rate (top) and mean placement error (bottom) against cube mass, one line per policy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, results in report.items():
        masses = [r.mass_kg for r in results]
        top.errorbar(masses, [100 * r.success_rate for r in results], yerr=[100 * r.seed_std for r in results],
                     marker="o", capsize=3, label=name)
        bottom.plot(masses, [100 * r.mean_error_m for r in results], marker="o", label=name)
    bottom.axhline(100 * PlaceConfig().place_tolerance_m, color="red", linestyle="--", label="tolerance")
    top.set(ylabel="success rate (%)", title="Pick-and-place vs. cube mass (bars: std across seeds)")
    bottom.set(xlabel="cube mass (kg)", ylabel="mean placement error (cm)")
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("saved", path)

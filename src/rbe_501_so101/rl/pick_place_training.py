"""Fresh, versioned full-task runs with exact environment and source reconstruction."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rbe_501_so101.config import PROJECT_ROOT, TrainConfig, resolve_project_path
from rbe_501_so101.control.carry_place import TransferConfig
from rbe_501_so101.control.supported_place import SupportSearchConfig
from rbe_501_so101.control.timed_oracle_acquisition import TimingConfig
from rbe_501_so101.control.tracked_place import PlacementTrackingConfig
from rbe_501_so101.envs.pick_place_config import OBSERVATION_FIELDS, PickPlaceConfig, PlacementRewardConfig
from rbe_501_so101.envs.pick_place_env import SO101PickPlaceEnv
from rbe_501_so101.rl.calibrated_pick_lift import canonical
from rbe_501_so101.rl.cartesian_zero_training import decode as decode_base, snapshot as base_snapshot
from rbe_501_so101.rl.pick_lift import NormalizedPolicy


def input_manifest(cfg: PickPlaceConfig) -> dict[str, str]:
    """Fingerprint runtime, model assets and calibration; normalize line endings only for text."""
    model = cfg.preset.environment().sim.model_path
    paths = set((PROJECT_ROOT / "src/rbe_501_so101").rglob("*.py"))
    paths.update(p for p in model.parent.rglob("*") if p.is_file())
    paths.add(resolve_project_path(cfg.preset.base.calibration_file))
    result = {}
    for path in sorted(paths):
        data = path.read_bytes()
        if path.suffix.lower() in (".py", ".xml", ".json"):
            data = data.replace(b"\r\n", b"\n")
        result[path.relative_to(PROJECT_ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
    return result


def snapshot(cfg: PickPlaceConfig) -> object:
    """Freeze both effective old pickup settings and new full-task semantics."""
    return canonical({"task": asdict(cfg), "pickup": base_snapshot(cfg.preset),
                      "observations": OBSERVATION_FIELDS, "action_shape": [3],
                      "authority": "<=1mm absolute Cartesian residual during grasped carry/lower; no gripper control; supported Z masked",
                      "success": "lifted, released, retreated, settled, supported, ungrasped, destination/height/linear/angular speed limits",
                      "hash_scheme": "sha256; LF-normalized .py/.xml/.json; binary assets unchanged"})


def decode(raw: dict) -> PickPlaceConfig:
    """Restore nested immutable dataclasses and tuple-valued yaw settings from JSON."""
    return PickPlaceConfig(**{**raw, "preset": decode_base(raw["preset"]),
        "transfer": TransferConfig(**raw["transfer"]), "support": SupportSearchConfig(**raw["support"]),
        "tracking": PlacementTrackingConfig(**raw["tracking"]), "timing": TimingConfig(**raw["timing"]),
        "reward": PlacementRewardConfig(**raw["reward"]), "yaw_offsets_deg": tuple(raw["yaw_offsets_deg"])})


@dataclass(frozen=True)
class PickPlaceRepository:
    """Own a new full-task checkpoint directory; never reuse old reaching or pickup policies."""

    root: Path

    @property
    def path(self) -> Path:
        """Full environment/training configuration beside the checkpoint companions."""
        return self.root / "run_config.json"

    def create(self, cfg: PickPlaceConfig, training: TrainConfig, seed: int) -> None:
        """Require a fresh directory and matching potential-shaping/training discount."""
        if cfg.reward.gamma != training.gamma:
            raise ValueError("Reward shaping and training gamma must match")
        self.root.mkdir(parents=True, exist_ok=False)
        record = {"schema": 1, "preset": "pick_place_residual_v1", "environment": asdict(cfg),
                  "snapshot": snapshot(cfg), "training": asdict(training), "seed": seed,
                  "input_sha256": input_manifest(cfg)}
        self.path.write_text(json.dumps(canonical(record), indent=2), encoding="utf-8")

    def load(self) -> PickPlaceConfig:
        """Reject changed dimensions, configuration, implementation or model assets."""
        record = json.loads(self.path.read_text(encoding="utf-8"))
        if record["schema"] != 1 or record["preset"] != "pick_place_residual_v1":
            raise ValueError("Expected full pick/place v1 preset")
        cfg = decode(record["environment"])
        if record["snapshot"] != snapshot(cfg) or record["input_sha256"] != input_manifest(cfg):
            raise ValueError("Full-task configuration/source/model snapshot changed")
        return cfg


def factory(cfg: PickPlaceConfig, monitor: Path | None = None) -> Callable[[], gym.Env]:
    """Monitor true placement outcome and actual safety/residual interventions."""
    def create() -> gym.Env:
        """Construct an independent complete-task environment."""
        keys = ("is_success", "placement_error_m", "mass_kg", "robot_table_contact_steps", "mujoco_warning",
                "lifted", "released", "retreated", "residual_steps", "residual_rejections", "episode_steps")
        return Monitor(SO101PickPlaceEnv(cfg), filename=str(monitor) if monitor else None, info_keywords=keys)
    return create


def load_policy(repo: PickPlaceRepository, seed: int = 0) -> NormalizedPolicy:
    """Require the full-task policy's own 72-value normalization statistics and preset."""
    vector = DummyVecEnv([factory(repo.load())])
    folder = repo.root / f"seed_{seed}"
    normalizer = VecNormalize.load(str(folder / "vecnormalize.pkl"), vector)
    normalizer.training, normalizer.norm_reward = False, False
    return NormalizedPolicy(SAC.load(folder / "model.zip", device="cpu"), normalizer)

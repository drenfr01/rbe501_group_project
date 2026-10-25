"""Unit tests for experiment presets, the checkpoint repository, and model construction."""

import json

import pytest
from stable_baselines3 import SAC

from rbe_501_so101.config import (
    PRESETS,
    ExperimentPreset,
    ReachTaskConfig,
    RewardConfig,
    SacTuning,
    TrainConfig,
)
from rbe_501_so101.rl.checkpoints import CheckpointRepository
from rbe_501_so101.rl.cli import parse_args
from rbe_501_so101.rl.evaluation import make_eval_env
from rbe_501_so101.rl.training import build_model, make_vec_env

SEED = 0


def test_legacy_preset_matches_defaults():
    assert PRESETS["legacy"] == ExperimentPreset()
    assert ExperimentPreset().task == ReachTaskConfig()
    assert ExperimentPreset().reward == RewardConfig()


def test_new_presets_share_parallel_sac_settings():
    tunings = [PRESETS[name].sac for name in ("control_long", "shaped", "residual_ik")]
    assert len({(t.n_envs, t.gradient_steps, t.buffer_size) for t in tunings}) == 1
    assert tunings[0].n_envs > 1
    assert tunings[0].target_entropy == "auto"
    assert tunings[1].target_entropy == tunings[2].target_entropy != "auto"


def test_residual_preset_uses_residual_actions_and_shaping():
    preset = PRESETS["residual_ik"]
    assert preset.task.action_mode == "residual_ik"
    assert preset.task.fine_error_obs
    assert preset.reward.velocity_penalty > 0 and preset.reward.precision_weight > 0


def test_preset_round_trips_through_dict():
    preset = PRESETS["shaped"]
    assert ExperimentPreset.from_dict(json.loads(json.dumps(preset.to_dict()))) == preset


def test_repository_saves_and_loads_preset(tmp_path):
    repo = CheckpointRepository(tmp_path, TrainConfig())
    repo.save_preset(SEED, "residual_ik", PRESETS["residual_ik"])
    assert repo.load_preset(SEED) == PRESETS["residual_ik"]
    assert repo.load_preset_name(SEED) == "residual_ik"


def test_repository_defaults_to_legacy_without_run_config(tmp_path):
    repo = CheckpointRepository(tmp_path, TrainConfig())
    assert repo.load_preset(SEED) == PRESETS["legacy"]


def test_repository_refuses_to_overwrite_existing_model(tmp_path):
    cfg = TrainConfig()
    repo = CheckpointRepository(tmp_path, cfg)
    repo.seed_dir(SEED).mkdir(parents=True)
    repo.model_path(SEED).write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        repo.ensure_untrained(SEED)


def test_make_vec_env_builds_n_envs_and_monitor(tmp_path):
    preset = ExperimentPreset(sac=SacTuning(n_envs=3))
    env = make_vec_env(preset, randomize=False, monitor_path=tmp_path / "train", gamma=0.98)
    try:
        assert env.num_envs == 3
        env.reset()
    finally:
        env.close()
    assert (tmp_path / "train.monitor.csv").exists()


def test_build_model_applies_sac_tuning(tmp_path):
    tuning = SacTuning(n_envs=2, gradient_steps=2, buffer_size=1000, target_entropy=-10.0)
    preset = ExperimentPreset(sac=tuning)
    env = make_vec_env(preset, randomize=False, monitor_path=None, gamma=0.98)
    try:
        model = build_model("sac", env, SEED, TrainConfig(), tuning, device="cpu")
        assert isinstance(model, SAC)
        assert model.gradient_steps == 2
        assert model.buffer_size == 1000
        assert model.target_entropy == pytest.approx(-10.0)
    finally:
        env.close()


def test_eval_env_matches_preset_observation_size():
    env = make_eval_env(PRESETS["residual_ik"], randomize=False)
    try:
        assert env.layout.include_joint_goal and env.layout.include_fine_error
        assert env.observation_space.shape == (env.layout.size,)
    finally:
        env.close()


def test_parse_args_preset_defaults_to_legacy():
    assert parse_args(TrainConfig(), []).preset == "legacy"
    assert parse_args(TrainConfig(), ["--preset", "shaped"]).preset == "shaped"

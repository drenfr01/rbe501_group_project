"""Tests for pick-and-lift RL plumbing: residual wrapper, training, mass-sweep evaluation, CLI."""

import json

import numpy as np
import pytest
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from rbe_501_so101.config import (
    ObjectMassConfig,
    ObjectConfig,
    PickLiftEvalConfig,
    PickLiftTaskConfig,
    ResidualConfig,
    TrainConfig,
)
from rbe_501_so101.control import ScriptedPickLift
from rbe_501_so101.rl.pick_lift import (
    MassResult,
    ResidualPickLift,
    RunInfo,
    evaluate_over_masses,
    evaluate_run,
    load_run,
    make_pick_lift_env,
    pick_lift_factory,
    train_pick_lift_seed,
)
from rbe_501_so101.rl.pick_lift_cli import parse_args
from rbe_501_so101.rl.checkpoint import TrainingCheckpoint

LIGHT = ObjectMassConfig(mode="fixed", fixed_kg=0.05)
SMOKE_TIMESTEPS = 64
SEED = 0


def test_zero_residual_reproduces_scripted_controller():
    residual_env = make_pick_lift_env(LIGHT, "residual")
    direct_env = make_pick_lift_env(LIGHT, "direct")
    controller = ScriptedPickLift.for_env(direct_env)
    residual_obs, _ = residual_env.reset(seed=SEED)
    direct_obs, _ = direct_env.reset(seed=SEED)
    controller.reset()
    zero = np.zeros(residual_env.action_space.shape, dtype=np.float32)
    for _ in range(120):
        residual_obs, *_ = residual_env.step(zero)
        direct_obs, *_ = direct_env.step(controller(direct_obs))
    np.testing.assert_allclose(residual_obs, direct_obs, atol=1e-6)
    residual_env.close()
    direct_env.close()


def test_residual_adds_scaled_policy_action():
    env = make_pick_lift_env(LIGHT, "residual")
    assert isinstance(env, ResidualPickLift)
    env.reset(seed=SEED)
    base = env.base_action()
    env.step(np.full(env.action_space.shape, -1.0, dtype=np.float32))
    expected = np.clip(base - ResidualConfig().scale, -1.0, 1.0)
    np.testing.assert_allclose(env.last_executed_action, expected, atol=1e-6)
    env.close()


def test_factory_monitors_success_and_mass(tmp_path):
    env = pick_lift_factory(LIGHT, "residual", tmp_path / "train")()
    assert isinstance(env, Monitor)
    assert {"is_success", "mass_kg", "lift_m"} <= set(env.info_keywords)
    env.close()


def test_scripted_sweep_succeeds_on_light_mass():
    eval_cfg = PickLiftEvalConfig(masses_kg=(0.05,), episodes_per_mass=2)
    results = evaluate_over_masses(lambda env: ScriptedPickLift.for_env(env), "direct", eval_cfg)
    assert results == [MassResult(mass_kg=0.05, success_rate=1.0, mean_lift_m=results[0].mean_lift_m, episodes=2)]
    assert results[0].mean_lift_m > 0.04


def test_train_saves_model_normalizer_and_run_info(tmp_path):
    run = RunInfo(mass=ObjectMassConfig(mode="random"), control="residual", algo="sac")
    seed_dir = train_pick_lift_seed(SEED, run, SMOKE_TIMESTEPS, tmp_path, TrainConfig(), device="cpu")
    cfg = TrainConfig()
    assert (seed_dir / cfg.model_filename).exists() and (seed_dir / cfg.vecnormalize_filename).exists()
    assert json.loads((tmp_path / "run_info.json").read_text())["mass"]["mode"] == "random"
    assert load_run(tmp_path) == run


def test_trained_policy_runs_in_mass_sweep(tmp_path):
    run = RunInfo(mass=LIGHT, control="residual", algo="sac")
    train_pick_lift_seed(SEED, run, SMOKE_TIMESTEPS, tmp_path, TrainConfig(), device="cpu")
    eval_cfg = PickLiftEvalConfig(masses_kg=(0.05, 1.0), episodes_per_mass=1)
    results = evaluate_run(tmp_path, eval_cfg, TrainConfig())
    assert [r.mass_kg for r in results] == [0.05, 1.0]
    assert all(0.0 <= r.success_rate <= 1.0 for r in results)


@pytest.mark.parametrize(
    ("argv", "mode", "dir_name"),
    [
        (["train", "--mass-mode", "fixed", "--mass-kg", "0.3"], "fixed", "pick_lift_fixed_300g_25mm"),
        (["train", "--mass-mode", "random", "--mass-range", "0.1", "1.5"], "random", "pick_lift_random_100_1500g_25mm"),
    ],
)
def test_cli_builds_mass_config_and_default_dir(argv, mode, dir_name):
    settings = parse_args(TrainConfig(), argv)
    assert settings.run.mass.mode == mode
    assert settings.model_dir.name == dir_name
    if mode == "fixed":
        assert settings.run.mass.fixed_kg == 0.3
    else:
        assert settings.run.mass.range_kg == (0.1, 1.5)


def test_training_rejects_different_mass_config_for_existing_run(tmp_path):
    """A saved policy cannot silently continue under a new object mass setting."""
    old = RunInfo(mass=ObjectMassConfig(mode="random", range_kg=(0.02, 2.0)), control="residual", algo="sac")
    old.save(tmp_path)
    new = RunInfo(mass=ObjectMassConfig(mode="random", range_kg=(0.005, 0.02)), control="residual", algo="sac")
    with pytest.raises(ValueError, match="different training settings"):
        train_pick_lift_seed(SEED, new, SMOKE_TIMESTEPS, tmp_path, TrainConfig(), device="cpu")
    assert load_run(tmp_path) == old


def test_training_rejects_different_cube_size_for_existing_run(tmp_path):
    """Changing geometry requires a new run, even if the masses stay the same."""
    old = RunInfo(mass=LIGHT, control="residual", algo="sac", object=ObjectConfig(side_m=0.03))
    old.save(tmp_path)
    new = RunInfo(mass=LIGHT, control="residual", algo="sac")
    with pytest.raises(ValueError, match="different training settings"):
        train_pick_lift_seed(SEED, new, SMOKE_TIMESTEPS, tmp_path, TrainConfig(), device="cpu")
    assert load_run(tmp_path) == old


def test_legacy_run_info_retains_original_cube_size(tmp_path):
    """Runs saved before cube size tracking still evaluate with their 30 mm cube."""
    (tmp_path / "run_info.json").write_text(
        json.dumps({"mass": {"mode": "fixed", "fixed_kg": 0.01, "range_kg": [0.005, 0.02]},
                    "control": "residual", "algo": "sac"})
    )
    assert load_run(tmp_path).object.side_m == 0.03


def test_env_uses_recorded_cube_size():
    """A policy's environment uses the cube geometry stored with that run."""
    object_cfg = ObjectConfig(side_m=0.03)
    env = make_pick_lift_env(LIGHT, "direct", object_cfg=object_cfg)
    assert env.config.object.side_m == object_cfg.side_m
    np.testing.assert_allclose(env.backend.model.geom(object_cfg.geom).size, object_cfg.side_m / 2)
    env.close()


def _trained_steps(seed_dir, cfg: TrainConfig) -> int:
    """Timesteps recorded in the final saved model of one seed."""
    return SAC.load(seed_dir / cfg.model_filename, device="cpu").num_timesteps


def test_train_writes_resumable_checkpoint(tmp_path):
    cfg = TrainConfig()
    run = RunInfo(mass=LIGHT, control="residual", algo="sac")
    seed_dir = train_pick_lift_seed(SEED, run, SMOKE_TIMESTEPS, tmp_path, cfg, device="cpu")
    checkpoint = TrainingCheckpoint.in_dir(seed_dir, cfg)
    assert checkpoint.exists()
    assert checkpoint.model.exists() and checkpoint.vecnormalize.exists() and checkpoint.replay_buffer.exists()


def test_rerun_resumes_from_checkpoint_up_to_new_total(tmp_path):
    cfg = TrainConfig()
    run = RunInfo(mass=LIGHT, control="residual", algo="sac")
    train_pick_lift_seed(SEED, run, SMOKE_TIMESTEPS, tmp_path, cfg, device="cpu")
    seed_dir = train_pick_lift_seed(SEED, run, 2 * SMOKE_TIMESTEPS, tmp_path, cfg, device="cpu")
    assert _trained_steps(seed_dir, cfg) == 2 * SMOKE_TIMESTEPS


def test_rerun_at_reached_total_does_not_train_further(tmp_path):
    cfg = TrainConfig()
    run = RunInfo(mass=LIGHT, control="residual", algo="sac")
    train_pick_lift_seed(SEED, run, SMOKE_TIMESTEPS, tmp_path, cfg, device="cpu")
    seed_dir = train_pick_lift_seed(SEED, run, SMOKE_TIMESTEPS, tmp_path, cfg, device="cpu")
    assert _trained_steps(seed_dir, cfg) == SMOKE_TIMESTEPS


def test_periodic_checkpoint_saves_during_training(tmp_path):
    cfg = TrainConfig(checkpoint_freq_steps=SMOKE_TIMESTEPS // 2)
    run = RunInfo(mass=LIGHT, control="residual", algo="sac")
    seed_dir = train_pick_lift_seed(SEED, run, SMOKE_TIMESTEPS, tmp_path, cfg, device="cpu")
    saves = json.loads(TrainingCheckpoint.in_dir(seed_dir, cfg).log.read_text())
    assert saves == [SMOKE_TIMESTEPS // 2, SMOKE_TIMESTEPS]


def test_resumed_run_appends_to_monitor_log(tmp_path):
    cfg = TrainConfig()
    run = RunInfo(mass=LIGHT, control="residual", algo="sac")
    episode = PickLiftTaskConfig().max_episode_steps
    train_pick_lift_seed(SEED, run, episode, tmp_path, cfg, device="cpu")
    seed_dir = train_pick_lift_seed(SEED, run, 2 * episode, tmp_path, cfg, device="cpu")
    lines = (seed_dir / "train.monitor.csv").read_text().splitlines()
    assert sum(line.startswith("#") for line in lines) == 1
    assert len(lines) == 2 + 2

"""Unit tests for evaluation helpers that do not require trained checkpoints."""

import numpy as np
import pytest

from rbe_501_so101.config import EnvOptions, RobustnessConfig, TrainConfig
from rbe_501_so101.rl.cli import parse_args
from rbe_501_so101.rl.evaluation import (
    EpisodeStats,
    EvalOverrides,
    PolicyReport,
    completed_episodes,
    passes,
    run_episode,
    summarize,
)
from rbe_501_so101.rl.policies import RandomPolicy


def _episode(error_m: float, speed: float = 0.01) -> EpisodeStats:
    """One-step holding window at a constant error."""
    return EpisodeStats(holding_errors_m=np.array([error_m]), mean_joint_speed=speed)


def test_completed_episodes_drops_none_and_requires_one_finished():
    done = completed_episodes([_episode(0.01), None, _episode(0.02)])
    assert len(done) == 2
    with pytest.raises(RuntimeError, match="No complete evaluation episodes"):
        completed_episodes([None, None])


def test_summarize_reports_millimeters():
    report = summarize({"0": [_episode(0.004)], "1": [_episode(0.006)]})
    assert report.mean_mm == pytest.approx(5.0)
    assert report.per_seed_mean_mm["0"] == pytest.approx(4.0)
    assert report.per_seed_mean_mm["1"] == pytest.approx(6.0)


def test_passes_requires_mean_p95_and_stillness():
    cfg = TrainConfig()
    trained = PolicyReport(mean_mm=3.0, p95_mm=6.0, per_seed_mean_mm={"0": 3.0}, mean_joint_speed_rad_s=0.01)
    ik_hold = PolicyReport(mean_mm=10.0, p95_mm=15.0, per_seed_mean_mm={"all": 10.0}, mean_joint_speed_rad_s=0.0)
    assert passes(trained, ik_hold, cfg)
    too_fast = PolicyReport(mean_mm=3.0, p95_mm=6.0, per_seed_mean_mm={"0": 3.0}, mean_joint_speed_rad_s=0.2)
    assert not passes(too_fast, ik_hold, cfg)


def test_parse_args_returns_run_settings_not_namespace():
    settings = parse_args(TrainConfig(), ["--eval", "--seeds", "0", "1", "--num-episodes", "4"])
    assert settings.evaluate is True
    assert settings.seeds == (0, 1)
    assert settings.num_episodes == 4
    assert settings.cpu is False


def test_parse_args_cpu_flag():
    settings = parse_args(TrainConfig(), ["--cpu"])
    assert settings.cpu is True


def test_parse_args_calibration_and_perturb_flags(tmp_path):
    settings = parse_args(TrainConfig(), ["--eval", "--calibration", str(tmp_path / "c.json"), "--perturb"])
    assert settings.overrides == EvalOverrides(calibration_path=tmp_path / "c.json", perturb=True)
    assert parse_args(TrainConfig(), ["--eval"]).overrides == EvalOverrides()


def test_eval_overrides_layer_on_preset_options(tmp_path):
    base = EnvOptions()
    assert EvalOverrides().apply(base) == base
    applied = EvalOverrides(calibration_path=tmp_path / "c.json", perturb=True).apply(base)
    assert applied.calibration_path == str(tmp_path / "c.json")
    assert applied.perturbation.delay_steps == max(RobustnessConfig().delay_steps)
    assert applied.perturbation.quantization_rad == RobustnessConfig().quantization_rad


def test_episode_stats_record_holding_actions(reach_env):
    policy = RandomPolicy(0, reach_env.layout.n_arm)
    stats = run_episode(reach_env, policy, seed=0, window=TrainConfig().holding_window_steps)
    assert stats.holding_actions.shape == (TrainConfig().holding_window_steps, reach_env.layout.n_arm)

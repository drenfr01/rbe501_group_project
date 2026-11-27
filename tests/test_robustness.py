"""Phase 2 robustness: saturation, decision rule, classification, and a 1-point sweep."""

from dataclasses import replace

import numpy as np
import pytest

from rbe_501_so101.config import PRESETS, RobustnessConfig, SacTuning, TrainConfig
from rbe_501_so101.rl.checkpoints import CheckpointRepository
from rbe_501_so101.rl.robustness import (
    TRAINED,
    SweepResult,
    classify_correction,
    slowed_preset,
    run_stiffness_sweep,
    saturation_fraction,
    transfer_decision,
    working_range,
)
from rbe_501_so101.rl.training import build_model, make_vec_env

TINY_TIMESTEPS = 120
TINY_BUFFER = 1_000


def _result(controller: str, value: float, mean_mm: float, sat: float | None = None, seeds=None) -> SweepResult:
    """Hand-built sweep point with p95 = 2 * mean."""
    per_seed = seeds or {"all": mean_mm}
    return SweepResult(controller, "kp_scale", value, mean_mm, 2 * mean_mm, per_seed, sat)


def test_saturation_fraction_counts_steps_with_any_large_component():
    actions = np.array([[0.1, 0.2], [0.96, 0.0], [0.0, -0.99], [0.5, 0.5]])
    assert saturation_fraction(actions, 0.95) == pytest.approx(0.5)


def test_transfer_decision_accepts_accurate_unsaturated_policy():
    results = [_result(TRAINED, 1.0, 1.5, 0.01, {"0": 1.4, "1": 1.6}), _result("ik_hold", 1.0, 10.0)]
    decision = transfer_decision(results, RobustnessConfig())
    assert decision.transfer
    assert len(decision.reasons) == 3


def test_transfer_decision_rejects_inaccurate_or_saturated_policy():
    cfg = RobustnessConfig()
    inaccurate = [_result(TRAINED, 1.0, 3.0, 0.0), _result("ik_hold", 1.0, 10.0)]
    saturated = [_result(TRAINED, 1.0, 1.0, 0.2), _result("ik_hold", 1.0, 10.0)]
    no_better = [_result(TRAINED, 1.0, 1.0, 0.0, {"0": 0.5, "1": 9.0}), _result("ik_hold", 1.0, 1.2)]
    assert not transfer_decision(inaccurate, cfg).transfer
    assert not transfer_decision(saturated, cfg).transfer
    assert not transfer_decision(no_better, cfg).transfer


def _sweep(rl_means: list[float]) -> list[SweepResult]:
    """Integral flat, gravity comp growing, RL as given, over scales 0.5/1/2."""
    scales = (0.5, 1.0, 2.0)
    results = [_result("integral", s, 1.0) for s in scales]
    results += [_result("gravity_comp", s, m) for s, m in zip(scales, (10.0, 1.0, 5.0), strict=True)]
    results += [_result(TRAINED, s, m, 0.0) for s, m in zip(scales, rl_means, strict=True)]
    return results


def test_flat_policy_is_classified_as_feedback():
    assert classify_correction(_sweep([1.1, 1.0, 1.2])).kind == "feedback"


def test_growing_policy_is_classified_as_feedforward():
    assert classify_correction(_sweep([9.0, 1.0, 4.0])).kind == "feedforward"


def test_working_range_is_contiguous_around_nominal():
    assert working_range(_sweep([3.0, 1.0, 1.5]), RobustnessConfig()) == (1.0, 2.0)
    assert working_range(_sweep([3.0, 3.0, 1.5]), RobustnessConfig()) is None


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> CheckpointRepository:
    """A barely trained residual_ik SAC seed, enough to exercise the sweep plumbing."""
    cfg = TrainConfig()
    repo = CheckpointRepository(tmp_path_factory.mktemp("ckpt"), cfg)
    preset = replace(PRESETS["residual_ik"], sac=SacTuning(buffer_size=TINY_BUFFER))
    repo.save_preset(0, "residual_ik", preset)
    env = make_vec_env(preset, randomize=False, monitor_path=None, gamma=cfg.gamma)
    model = build_model("sac", env, 0, cfg, preset.sac, device="cpu")
    model.learn(total_timesteps=TINY_TIMESTEPS)
    model.save(repo.model_path(0))
    env.save(str(repo.normalizer_path(0)))
    env.close()
    return repo


def test_one_point_sweep_returns_every_controller_with_finite_errors(tiny_checkpoint):
    cfg = RobustnessConfig(kp_scales=(0.75,), episodes_per_point=1)
    results = run_stiffness_sweep(tiny_checkpoint, (0,), cfg)
    assert {r.controller for r in results} == {TRAINED, "ik_hold", "gravity_comp", "integral"}
    assert all(r.parameter == "kp_scale" and r.value == 0.75 for r in results)
    assert all(np.isfinite(r.mean_mm) and np.isfinite(r.p95_mm) for r in results)
    trained = next(r for r in results if r.controller == TRAINED)
    assert trained.saturation_fraction is not None and 0.0 <= trained.saturation_fraction <= 1.0


def test_slowed_preset_scales_only_the_rate_limit():
    preset = PRESETS["residual_ik"]
    slow = slowed_preset(preset, 0.5)
    assert slow.task.delta_q_max_rad == pytest.approx(0.5 * preset.task.delta_q_max_rad)
    assert slow.task.residual_max_rad == preset.task.residual_max_rad
    assert slowed_preset(preset, 1.0) is preset

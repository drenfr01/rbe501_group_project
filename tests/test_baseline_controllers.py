"""Gravity-compensation (feedforward) and joint-integral (feedback) baselines in sim."""

import numpy as np
import pytest

from rbe_501_so101.config import (
    PRESETS,
    DynamicsRandomizationConfig,
    EnvOptions,
    IntegralControlConfig,
    ReachTaskConfig,
    SimConfig,
    TrainConfig,
)
from rbe_501_so101.envs.factory import make_reach_env
from rbe_501_so101.envs.types import ObservationLayout
from rbe_501_so101.rl.evaluation import evaluate
from rbe_501_so101.rl.policies import GravityCompPolicy, IKHoldPolicy, JointIntegralPolicy, make_baseline

SIM = SimConfig()
TASK = ReachTaskConfig()
LAYOUT = ObservationLayout(n_arm=len(SIM.arm_joints))
EPISODES = 3
EVAL_CFG = TrainConfig()


def _fixed_kp(scale: float) -> EnvOptions:
    """Env options that scale every actuator's stiffness by exactly `scale`."""
    randomization = DynamicsRandomizationConfig(
        enabled=True, kp_scale_range=(scale, scale), damping_scale_range=(1.0, 1.0), payload_mass_range_kg=(0.0, 0.0)
    )
    return EnvOptions(randomization=randomization)


def _mean_error_mm(name: str, kp_scale: float) -> float:
    """Mean holding error of one baseline over a few shared targets."""
    env = make_reach_env(PRESETS["legacy"], randomize=False, options=_fixed_kp(kp_scale))
    policy = make_baseline(name, SIM, TASK, LAYOUT)
    try:
        stats = evaluate(env, policy, EPISODES, EVAL_CFG)
    finally:
        env.close()
    return 1000.0 * float(np.mean([ep.holding_errors_m for ep in stats]))


@pytest.fixture(scope="module")
def errors() -> dict[tuple[str, float], float]:
    """Mean error for every controller at nominal and half stiffness."""
    names = ("ik_hold", "gravity_comp", "integral")
    return {(n, s): _mean_error_mm(n, s) for n in names for s in (1.0, 0.5)}


def test_make_baseline_returns_expected_types():
    assert isinstance(make_baseline("ik_hold", SIM, TASK, LAYOUT), IKHoldPolicy)
    assert isinstance(make_baseline("gravity_comp", SIM, TASK, LAYOUT), GravityCompPolicy)
    assert isinstance(make_baseline("integral", SIM, TASK, LAYOUT, integral=IntegralControlConfig()), JointIntegralPolicy)


def test_both_corrections_beat_ik_hold_in_nominal_sim(errors):
    assert errors[("gravity_comp", 1.0)] < errors[("ik_hold", 1.0)]
    assert errors[("integral", 1.0)] < errors[("ik_hold", 1.0)]


def test_softer_motors_separate_feedback_from_feedforward(errors):
    assert errors[("integral", 0.5)] < errors[("ik_hold", 0.5)]
    assert errors[("gravity_comp", 0.5)] > errors[("gravity_comp", 1.0)]


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError):
        make_baseline("pid", SIM, TASK, LAYOUT)

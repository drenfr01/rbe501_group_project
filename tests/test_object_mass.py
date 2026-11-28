"""Unit tests for object-mass sampling: the fixed vs. randomized training parameter."""

import numpy as np
import pytest

from rbe_501_so101.config import ObjectMassConfig
from rbe_501_so101.config import PickLiftEvalConfig
from rbe_501_so101.envs.pick_lift_backend import cube_inertia

N_DRAWS = 200


def test_default_mass_settings_match_printed_cubes():
    """Training and evaluation defaults cover the 5–20 g printed cubes."""
    mass = ObjectMassConfig()
    assert mass.fixed_kg == 0.01
    assert mass.range_kg == (0.005, 0.02)
    assert PickLiftEvalConfig().masses_kg == (0.005, 0.01, 0.015, 0.02)


def test_fixed_mode_always_returns_fixed_mass(rng):
    mass = ObjectMassConfig(mode="fixed", fixed_kg=0.12)
    assert {mass.sample(rng) for _ in range(N_DRAWS)} == {0.12}


def test_random_mode_draws_inside_range(rng):
    low, high = 0.02, 2.0
    mass = ObjectMassConfig(mode="random", range_kg=(low, high))
    draws = np.array([mass.sample(rng) for _ in range(N_DRAWS)])
    assert np.all((draws >= low) & (draws <= high))
    assert draws.std() > 0.1 * (high - low)


def test_random_mode_is_reproducible_with_same_seed():
    mass = ObjectMassConfig(mode="random")
    first_rng = np.random.default_rng(7)
    second_rng = np.random.default_rng(7)
    first = [mass.sample(first_rng) for _ in range(3)]
    second = [mass.sample(second_rng) for _ in range(3)]
    assert first == second
    assert len(set(first)) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fixed_kg": 0.0},
        {"fixed_kg": -0.1},
        {"range_kg": (0.0, 0.5)},
        {"range_kg": (0.5, 0.1)},
    ],
)
def test_invalid_masses_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ObjectMassConfig(**kwargs)


def test_cube_inertia_matches_solid_cube_formula():
    mass, side = 0.3, 0.03
    expected = mass * side**2 / 6.0
    np.testing.assert_allclose(cube_inertia(mass, side), [expected] * 3)

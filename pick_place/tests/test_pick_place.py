"""Tests for pick-and-place: workspace and goals, reward rules, scripted controller, and RL plumbing."""

import json
import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rbe_501_so101.config import ObjectMassConfig, TrainConfig
from rbe_501_so101.envs.types import GraspState, ObjectState, RobotState
from rbe_501_so101.pick_place.env import (
    MassResult,
    PickPlaceReward,
    PlaceConfig,
    PlacePhase,
    PlaceRun,
    ScriptedPickPlace,
    SO101PickPlaceEnv,
    _pool,
    compare,
    episode_z_test,
    seed_permutation_p,
    significance,
    evaluate,
    make_env,
    trained_runs,
    train,
)
from rbe_501_so101.rl.pick_lift import RunInfo

LIGHT_KG, HEAVY_KG = 0.05, 0.3
FIXED_1KG = ObjectMassConfig(mode="fixed", fixed_kg=1.0)
N_JOINTS = 6
CFG = PlaceConfig()


def _env(mass_kg: float = LIGHT_KG) -> SO101PickPlaceEnv:
    """Pick-and-place env with a fixed cube mass."""
    return SO101PickPlaceEnv(mass_cfg=ObjectMassConfig(mode="fixed", fixed_kg=mass_kg))


def _xy(env: SO101PickPlaceEnv, radius: float, azimuth_deg: float) -> np.ndarray:
    """World XY at a radius and azimuth around the pan axis."""
    phi = np.deg2rad(azimuth_deg)
    return env.backend.pan_axis_xy + radius * np.array([np.cos(phi), np.sin(phi)])


def _scripted_episode(env: SO101PickPlaceEnv, seed: int, options: dict | None = None) -> dict:
    """Run the scripted controller for one episode; return the last info."""
    controller = ScriptedPickPlace.for_env(env)
    obs, _ = env.reset(seed=seed, options=options)
    controller.reset()
    truncated, info = False, {}
    while not truncated:
        obs, _, _, truncated, info = env.step(controller(obs))
    return info


@pytest.fixture
def env():
    """Light-cube pick-and-place env."""
    env = _env()
    yield env
    env.close()


# ---------------------------------------------------------------- workspace and goals


@pytest.mark.parametrize(("radius", "azimuth", "inside"), [(0.10, 0, True), (0.24, 90, True), (0.17, -90, True),
                                                           (0.05, 0, False), (0.30, 0, False), (0.17, 120, False)])
def test_workspace_covers_both_sides_up_to_90_deg(env, radius, azimuth, inside):
    assert env.contains(_xy(env, radius, azimuth)) == inside


def test_random_goal_is_reachable_and_away_from_cube(env):
    for seed in range(10):
        env.reset(seed=seed)
        p_obj = env.grasp_state.obj.p_s
        assert env.contains(p_obj[:2]) and env.contains(env.p_goal[:2])
        assert np.linalg.norm(env.p_goal[:2] - p_obj[:2]) >= CFG.min_separation_m


def test_user_start_and_goal_are_used_exactly(env):
    start, goal = _xy(env, 0.2, -70), _xy(env, 0.15, 80)
    obs, _ = env.reset(seed=0, options={"object_xy": start, "goal_xy": goal})
    np.testing.assert_allclose(env.grasp_state.obj.p_s[:2], start, atol=1e-9)
    np.testing.assert_allclose(env.p_goal, [*goal, env.backend.rest_z])
    np.testing.assert_allclose(obs[env.layout.goal_pos], env.p_goal, rtol=1e-6)
    np.testing.assert_allclose(env.backend.data.mocap_pos[env.backend.ids.target_mocap], env.p_goal)


@pytest.mark.parametrize(("options", "message"), [({"goal_xy": (0.3, 0.0)}, "outside the workspace"),
                                                  ({"goal": (0.0, 0.0)}, "unknown reset option")])
def test_bad_reset_options_are_rejected(env, options, message):
    with pytest.raises(ValueError, match=message):
        env.reset(seed=0, options=options)


def test_check_env_passes_without_warnings(env):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env, skip_render_check=True)


# ---------------------------------------------------------------- reward rules

REST_Z = 0.49
P_START, P_GOAL = np.array([0.0, 0.1, REST_Z]), np.array([0.0, -0.1, REST_Z])
UP = np.array([0.0, 0.0, 1.0])


def _grasp(p_obj: np.ndarray, grasped: bool = False, pinch_up: float = 0.0) -> GraspState:
    """Grasp state with the cube at p_obj and the pinch point `pinch_up` above it."""
    zeros = np.zeros(N_JOINTS)
    robot = RobotState(qpos=zeros, qvel=zeros, q_cmd=zeros, tcp_pos=p_obj)
    obj = ObjectState(p_s=np.asarray(p_obj, dtype=float), rest_z=REST_Z, mass_kg=LIGHT_KG)
    return GraspState(robot, p_obj + pinch_up * UP, obj, grasped)


def _staged(s: GraspState, p_release: np.ndarray | None = None) -> float:
    """Staged reward with zero actions."""
    zero = np.zeros(N_JOINTS)
    return PickPlaceReward(CFG)(s, 2 if s.grasped else 0, P_GOAL, p_release, zero, zero).staged


def test_reward_grows_through_every_stage():
    h = CFG.carry_height_m
    values = [
        _staged(_grasp(P_START, pinch_up=0.1)),  # reaching
        _staged(_grasp(P_START, grasped=True)),  # grasped
        _staged(_grasp(P_START + h * UP, grasped=True)),  # lifted
        _staged(_grasp((P_START + P_GOAL) / 2 + h * UP, grasped=True)),  # carrying
        _staged(_grasp(P_GOAL + h * UP, grasped=True)),  # above the goal
        _staged(_grasp(P_GOAL, grasped=True)),  # lowered
        _staged(_grasp(P_GOAL), p_release=P_GOAL),  # released
    ]
    assert np.all(np.diff(values) > 0), values
    assert values[-1] == CFG.success_weight


@pytest.mark.parametrize(
    ("cube", "grasped", "pads", "release", "success"),
    [
        (P_GOAL, False, 0, P_GOAL, True),  # placed and let go
        (P_GOAL, True, 2, None, False),  # still held
        (P_GOAL, False, 1, P_GOAL, False),  # one finger still touching
        (P_GOAL, False, 0, None, False),  # never grasped (pushed there)
        (P_GOAL, False, 0, P_START, False),  # let go elsewhere, then pushed
        (P_GOAL, False, 0, P_GOAL + 2 * CFG.max_release_height_m * UP, False),  # dropped from height
        (P_GOAL + [1.5 * CFG.place_tolerance_m, 0, 0], False, 0, P_GOAL, False),  # outside tolerance
        (P_GOAL + 2 * CFG.rest_tolerance_m * UP, False, 0, P_GOAL, False),  # not on the table
    ],
)
def test_success_rules(cube, grasped, pads, release, success):
    assert PickPlaceReward(CFG).is_success(_grasp(cube, grasped), pads, P_GOAL, release) == success


def test_total_is_staged_minus_action_and_smoothness_costs():
    reward = PickPlaceReward(PlaceConfig(smoothness_penalty=0.5))
    action = np.full(N_JOINTS, 0.5)
    terms = reward(_grasp(P_START, grasped=True), 2, P_GOAL, None, action, np.zeros(N_JOINTS))
    assert terms.total == pytest.approx(terms.staged - CFG.action_penalty * 1.5 - 0.5 * 1.5)


# ---------------------------------------------------------------- scripted controller (physics)


@pytest.mark.parametrize("seed", range(3))
def test_scripted_places_at_random_goals(seed):
    env = _env()
    info = _scripted_episode(env, seed)
    env.close()
    assert info["is_success"], info


@pytest.mark.parametrize(("start", "goal"), [((0.22, -85), (0.22, 85)), ((0.24, 0), (0.11, 0))])
def test_scripted_places_between_workspace_corners(start, goal):
    env = _env()
    info = _scripted_episode(env, 0, {"object_xy": _xy(env, *start), "goal_xy": _xy(env, *goal)})
    env.close()
    assert info["is_success"], info


def test_scripted_places_heavy_cube_and_ends_released():
    env = _env(HEAVY_KG)
    controller = ScriptedPickPlace.for_env(env)
    obs, _ = env.reset(seed=0)
    controller.reset()
    truncated = False
    while not truncated:
        action = controller(obs)
        assert np.all(np.abs(action) <= 1.0)
        obs, _, _, truncated, info = env.step(action)
    env.close()
    assert info["is_success"] and controller.phase == PlacePhase.RETREAT


# ---------------------------------------------------------------- training and evaluation

V1 = {"residual_scale": 0.5, "residual_penalty": 0.0, "arm_only": False, "tag": ""}


def test_zero_residual_reproduces_scripted_controller():
    mass = ObjectMassConfig(fixed_kg=LIGHT_KG)
    residual, direct = make_env(mass, PlaceRun(mass)), make_env(mass)
    controller = ScriptedPickPlace.for_env(direct)
    obs_r, _ = residual.reset(seed=0)
    obs_d, _ = direct.reset(seed=0)
    controller.reset()
    for _ in range(150):
        obs_r, *_ = residual.step(np.zeros(N_JOINTS, dtype=np.float32))
        obs_d, *_ = direct.step(controller(obs_d))
    np.testing.assert_allclose(obs_r, obs_d, atol=1e-6)
    residual.close()
    direct.close()


def test_v2_residual_is_small_arm_only_and_penalized():
    mass = ObjectMassConfig(fixed_kg=LIGHT_KG)
    run = PlaceRun(mass)
    env = make_env(mass, run)
    env.reset(seed=0)
    base = env.base_action()
    _, reward, _, _, info = env.step(np.ones(N_JOINTS, dtype=np.float32))
    executed = env.last_executed_action
    np.testing.assert_allclose(executed[:-1], np.clip(base[:-1] + run.residual_scale, -1, 1), atol=1e-6)
    assert executed[-1] == pytest.approx(base[-1])  # the gripper stays scripted
    assert info["residual_penalty"] == pytest.approx(run.residual_penalty * (N_JOINTS - 1))
    assert reward == pytest.approx(info["reward_total"] - info["residual_penalty"])
    env.close()


def test_v1_settings_reproduce_the_first_runs():
    mass = ObjectMassConfig(fixed_kg=LIGHT_KG)
    env = make_env(mass, PlaceRun(mass, **V1))
    env.reset(seed=0)
    base = env.base_action()
    _, reward, _, _, info = env.step(-np.ones(N_JOINTS, dtype=np.float32))
    np.testing.assert_allclose(env.last_executed_action, np.clip(base - 0.5, -1, 1), atol=1e-6)
    assert reward == info["reward_total"]
    env.close()


@pytest.mark.parametrize(("mode", "folder"), [("fixed", "pick_place_fixed_1000g_25mm_v2"),
                                              ("random", "pick_place_random_50_1000g_25mm_v2")])
def test_run_folder_names_mass_setting_and_version(mode, folder):
    assert PlaceRun(ObjectMassConfig(mode=mode, fixed_kg=1.0, range_kg=(0.05, 1.0))).name == folder


def test_first_runs_load_as_v1_in_their_own_folder(tmp_path):
    """run_info.json files from before v2 have no residual settings; they load as v1 and keep their folder name."""
    old = {"mass": {"mode": "fixed", "fixed_kg": 1.0, "range_kg": [0.05, 1.0]}, "control": "residual", "algo": "sac",
           "object": {"body": "cube", "geom": "cube_geom", "side_m": 0.025}, "task": "place"}
    (tmp_path / "run_info.json").write_text(json.dumps(old))
    run = PlaceRun.load(tmp_path)
    assert run == PlaceRun(ObjectMassConfig(mode="fixed", fixed_kg=1.0, range_kg=(0.05, 1.0)), **V1)
    assert run.name == "pick_place_fixed_1000g_25mm"
    with pytest.raises(ValueError, match="different training settings"):
        train(PlaceRun(FIXED_1KG), 0, 64, tmp_path, TrainConfig(), device="cpu")


def test_training_saves_a_resumable_place_run(tmp_path):
    run = PlaceRun(FIXED_1KG)
    seed_dir = train(run, 0, 64, tmp_path, TrainConfig(), device="cpu")
    assert (seed_dir / TrainConfig().model_filename).exists()
    assert json.loads((tmp_path / "run_info.json").read_text())["task"] == "place"
    assert PlaceRun.load(tmp_path) == run
    with pytest.raises(ValueError, match="different training settings"):
        train(PlaceRun(ObjectMassConfig(fixed_kg=0.5)), 0, 64, tmp_path, TrainConfig(), device="cpu")


def test_compare_rejects_pick_lift_runs(tmp_path):
    RunInfo(mass=ObjectMassConfig(), control="residual", algo="sac").save(tmp_path)
    with pytest.raises(ValueError, match="not pick-and-place"):
        compare([tmp_path], masses=(LIGHT_KG,), episodes=1)


def test_scripted_evaluation_reports_placement_error():
    [result] = evaluate(ScriptedPickPlace.for_env, None, masses=(LIGHT_KG,), episodes=2)
    assert result.success_rate == 1.0 and result.mean_error_m < CFG.place_tolerance_m


def test_compare_lists_only_trained_runs_of_the_current_version(tmp_path):
    for run in (PlaceRun(FIXED_1KG, **V1), PlaceRun(FIXED_1KG), PlaceRun(ObjectMassConfig(fixed_kg=0.5))):
        run.save(tmp_path / run.name)
    for run in (PlaceRun(FIXED_1KG, **V1), PlaceRun(FIXED_1KG)):
        (tmp_path / run.name / "seed_0").mkdir()
        (tmp_path / run.name / "seed_0" / TrainConfig().model_filename).touch()
    assert trained_runs("v2", tmp_path) == [tmp_path / PlaceRun(FIXED_1KG).name]


# ---------------------------------------------------------------- statistics across seeds


def _result(rate, episodes=50, mass=1.0, seeds=()):
    return MassResult(mass, rate, 0.01, episodes, seeds or (rate,))


def test_pooling_keeps_each_seeds_success_rate():
    pooled = _pool((_result(0.8), _result(0.6), _result(0.7)))
    assert pooled.seed_rates == (0.8, 0.6, 0.7)
    assert pooled.success_rate == pytest.approx(0.7) and pooled.episodes == 150
    assert pooled.seed_std == pytest.approx(0.1)


def test_episode_z_test_matches_the_two_proportion_formula():
    z, p = episode_z_test(_result(0.69, 150), _result(0.59, 150))
    assert z == pytest.approx(1.80, abs=0.01) and 0.05 < p < 0.08
    assert episode_z_test(_result(0.5, 100), _result(0.5, 100)) == (0.0, 1.0)


def test_seed_permutation_p_is_exact():
    assert seed_permutation_p((0.9,) * 5, (0.9,) * 5) == 1.0
    assert seed_permutation_p((0.9, 0.91, 0.92, 0.93, 0.94), (0.5, 0.51, 0.52, 0.53, 0.54)) == pytest.approx(2 / 252)


def test_significance_compares_the_two_trained_runs_at_every_mass():
    fixed = [_result(0.7, 150, m, (0.7, 0.72, 0.68)) for m in (1.0, 1.5)]
    random = [_result(0.6, 150, m, (0.58, 0.6, 0.62)) for m in (1.0, 1.5)]
    text = significance({"scripted": fixed, "fixed": fixed, "random": random})
    assert "fixed - random" in text and text.count("\n") == 2

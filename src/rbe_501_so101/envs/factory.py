"""One place that turns a preset plus env options into reach-env constructor arguments."""

from dataclasses import replace
from typing import Any

from rbe_501_so101.config import EnvOptions, ExperimentPreset, IKConfig, SimConfig, resolve_project_path
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend, RenderMode
from rbe_501_so101.envs.perturbed_backend import PerturbedBackend
from rbe_501_so101.envs.reach_env import SO101ReachEnv, task_home_deg
from rbe_501_so101.kinematics import PositionIKSolver


def reach_env_kwargs(
    preset: ExperimentPreset,
    randomize: bool,
    render_mode: RenderMode | None = None,
    options: EnvOptions | None = None,
) -> dict[str, Any]:
    """Keyword arguments for `SO101ReachEnv` (or `gym.make`).

    `options` defaults to the preset's own; randomization is on if either the
    caller or the options ask for it. A perturbation wraps the MuJoCo backend
    in `PerturbedBackend`, which the env then receives already built.
    """
    options = options or preset.env
    sim_cfg = _sim_config(options)
    randomization = replace(options.randomization, enabled=randomize or options.randomization.enabled)
    base = {"render_mode": render_mode, "task_cfg": preset.task, "reward_cfg": preset.reward}
    if not options.perturbation.active:
        return {**base, "sim_cfg": sim_cfg, "randomization_cfg": randomization}
    home_cfg = replace(sim_cfg, initial_deg=task_home_deg(sim_cfg, preset.task))
    inner = MujocoSo101Backend(home_cfg, randomization, render_mode)
    kinematics = PositionIKSolver(inner.model, inner.ids, sim_cfg, IKConfig())
    return {**base, "backend": PerturbedBackend(inner, options.perturbation, sim_cfg.control_dt, kinematics)}


def make_reach_env(
    preset: ExperimentPreset,
    randomize: bool,
    render_mode: RenderMode | None = None,
    options: EnvOptions | None = None,
) -> SO101ReachEnv:
    """Unwrapped reach env built from a preset and (optionally overridden) env options."""
    return SO101ReachEnv(**reach_env_kwargs(preset, randomize, render_mode, options))


def _sim_config(options: EnvOptions) -> SimConfig:
    """Default sim config plus the options' calibration file, if any."""
    if options.calibration_path is None:
        return SimConfig()
    return SimConfig(calibration_path=resolve_project_path(options.calibration_path))

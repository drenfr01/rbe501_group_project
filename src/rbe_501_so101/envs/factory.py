"""One place that turns a preset plus env options into reach-env constructor arguments."""

from dataclasses import replace
from typing import Any

import numpy as np

from rbe_501_so101.config import (
    EnvOptions,
    ExperimentPreset,
    IKConfig,
    ServoEffectsConfig,
    SimConfig,
    TargetSamplerConfig,
    resolve_project_path,
)
from rbe_501_so101.envs.calibration import ActuatorCalibration
from rbe_501_so101.envs.mujoco_backend import MujocoSo101Backend, RenderMode
from rbe_501_so101.envs.perturbed_backend import PerturbedBackend
from rbe_501_so101.envs.protocols import RobotBackend
from rbe_501_so101.envs.reach_env import SO101ReachEnv, task_home_deg
from rbe_501_so101.envs.servo_effects import ServoEffectsBackend, StictionParams
from rbe_501_so101.kinematics import PositionIKSolver
from rbe_501_so101.kinematics.gravity import GravityTorqueModel


def reach_env_kwargs(
    preset: ExperimentPreset,
    randomize: bool,
    render_mode: RenderMode | None = None,
    options: EnvOptions | None = None,
) -> dict[str, Any]:
    """Keyword arguments for `SO101ReachEnv` (or `gym.make`).

    `options` defaults to the preset's own; randomization is on if either the
    caller or the options ask for it. Servo effects wrap the MuJoCo backend in
    `ServoEffectsBackend` (what reaches the servo), and a perturbation wraps
    that in `PerturbedBackend` (what the controller sees); the env then
    receives the backend already built.
    """
    options = options or preset.env
    sim_cfg = _sim_config(options)
    randomization = replace(options.randomization, enabled=randomize or options.randomization.enabled)
    base = {
        "render_mode": render_mode,
        "task_cfg": preset.task,
        "reward_cfg": preset.reward,
        "sampler_cfg": replace(TargetSamplerConfig(), joint_window_rad=options.target_joint_window_rad),
    }
    if not (options.perturbation.active or options.servo.active):
        return {**base, "sim_cfg": sim_cfg, "randomization_cfg": randomization}
    home_cfg = replace(sim_cfg, initial_deg=task_home_deg(sim_cfg, preset.task))
    backend: RobotBackend = MujocoSo101Backend(home_cfg, randomization, render_mode)
    kinematics = PositionIKSolver(backend.model, backend.ids, sim_cfg, IKConfig())
    if options.servo.active:
        backend = servo_effects(backend, options.servo, sim_cfg, options.target_joint_window_rad)
    if options.perturbation.active:
        backend = PerturbedBackend(backend, options.perturbation, sim_cfg.control_dt, kinematics)
    return {**base, "backend": backend}


def servo_effects(
    inner: MujocoSo101Backend,
    cfg: ServoEffectsConfig,
    sim_cfg: SimConfig,
    command_window: tuple[tuple[float, float], ...] | None = None,
) -> ServoEffectsBackend:
    """Wrap a MuJoCo arm with the calibrated stiction band and/or LeRobot's relative clip (and command window)."""
    if cfg.stiction and sim_cfg.calibration_path is None:
        raise ValueError("stiction needs a calibration file with a fitted friction band")
    params = StictionParams(np.zeros(len(sim_cfg.arm_joints)), np.zeros(len(sim_cfg.arm_joints)))
    if sim_cfg.calibration_path is not None:
        params = StictionParams.from_calibration(ActuatorCalibration.load(sim_cfg.calibration_path), sim_cfg.arm_joints)
    return ServoEffectsBackend(inner, params, cfg, GravityTorqueModel(inner.model, inner.ids), command_window)


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

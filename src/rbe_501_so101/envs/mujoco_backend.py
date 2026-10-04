"""MuJoCo implementation of the RobotBackend protocol for the SO-101 arm."""

from dataclasses import dataclass
from typing import Literal, Self

import mujoco
import numpy as np

from rbe_501_so101.config import DynamicsRandomizationConfig, RenderConfig, SimConfig
from rbe_501_so101.envs.types import JointLimits, RobotState

RenderMode = Literal["human", "rgb_array"]
KP_GAIN_COL = 0
KP_BIAS_COL = 1
KV_BIAS_COL = 2


@dataclass(frozen=True)
class ModelIndices:
    """Pre-cached MuJoCo addresses so the stepping loop never does string lookups."""

    n_arm: int
    joint_ids: np.ndarray
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    actuator_ids: np.ndarray
    tcp_site: int
    target_mocap: int
    payload_body: int

    @classmethod
    def from_model(cls, model: mujoco.MjModel, cfg: SimConfig) -> Self:
        """Resolve all configured names to integer IDs and addresses."""
        joint_ids = np.array([model.joint(n).id for n in cfg.joints])
        return cls(
            n_arm=len(cfg.arm_joints),
            joint_ids=joint_ids,
            qpos_adr=model.jnt_qposadr[joint_ids].copy(),
            dof_adr=model.jnt_dofadr[joint_ids].copy(),
            actuator_ids=np.array([model.actuator(n + cfg.actuator_suffix).id for n in cfg.joints]),
            tcp_site=model.site(cfg.tcp_site).id,
            target_mocap=int(model.body(cfg.target_mocap_body).mocapid[0]),
            payload_body=model.body(cfg.payload_body).id,
        )


@dataclass(frozen=True)
class NominalDynamics:
    """Unrandomized actuator gains, joint damping, and payload body mass."""

    kp: np.ndarray
    kv_bias: np.ndarray
    dof_damping: np.ndarray
    payload_mass: float

    @classmethod
    def capture(cls, model: mujoco.MjModel, ids: ModelIndices) -> Self:
        """Copy the current model values for later restoration and scaling."""
        return cls(
            kp=model.actuator_gainprm[ids.actuator_ids, KP_GAIN_COL].copy(),
            kv_bias=model.actuator_biasprm[ids.actuator_ids, KV_BIAS_COL].copy(),
            dof_damping=model.dof_damping[ids.dof_adr].copy(),
            payload_mass=float(model.body_mass[ids.payload_body]),
        )


class MujocoSo101Backend:
    """Drives the simulated SO-101 with delta joint commands at the control rate.

    Each `step()` writes the commanded angles to the position actuators and runs
    `n_substeps` physics steps. Commands are integrated on the commanded angle
    (not the measured one), so a zero delta holds the command steady.
    """

    def __init__(
        self,
        sim_cfg: SimConfig = SimConfig(),
        randomization: DynamicsRandomizationConfig = DynamicsRandomizationConfig(),
        render_mode: RenderMode | None = None,
        render_cfg: RenderConfig = RenderConfig(),
    ) -> None:
        self._cfg = sim_cfg
        self._randomization = randomization
        self._render_mode = render_mode
        self._render_cfg = render_cfg
        self.model = mujoco.MjModel.from_xml_path(str(sim_cfg.model_path))
        self.data = mujoco.MjData(self.model)
        self._ids = ModelIndices.from_model(self.model, sim_cfg)
        self._apply_torque_caps()
        self._nominal = NominalDynamics.capture(self.model, self._ids)
        self._limits = self._read_limits()
        self._home = np.deg2rad(np.asarray(sim_cfg.initial_deg, dtype=float))
        self._q_cmd = self._home.copy()
        self._viewer = None
        self._renderer: mujoco.Renderer | None = None

    def reset(self, rng: np.random.Generator) -> RobotState:
        """Restore (or randomize) dynamics and place the arm at the home pose."""
        self._restore_nominal_dynamics()
        if self._randomization.enabled:
            self._randomize_dynamics(rng)
        mujoco.mj_resetData(self.model, self.data)
        self._q_cmd = self._home.copy()
        self.data.qpos[self._ids.qpos_adr] = self._q_cmd
        self.data.ctrl[self._ids.actuator_ids] = self._q_cmd
        mujoco.mj_forward(self.model, self.data)
        return self._state()

    def apply_delta(self, delta_arm_rad: np.ndarray) -> None:
        """Integrate a relative arm command and clip it to the joint limits."""
        n = self._ids.n_arm
        self._q_cmd[:n] = np.clip(self._q_cmd[:n] + delta_arm_rad, self._limits.lower, self._limits.upper)

    def step(self) -> RobotState:
        """Hold the current command for one control period and report the state."""
        self.data.ctrl[self._ids.actuator_ids] = self._q_cmd
        for _ in range(self._cfg.n_substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._state()

    def joint_limits(self) -> JointLimits:
        """Return the arm joint position limits in radians."""
        return self._limits

    def set_target_marker(self, xyz: np.ndarray) -> None:
        """Move the mocap target sphere and refresh derived positions."""
        self.data.mocap_pos[self._ids.target_mocap] = xyz
        mujoco.mj_forward(self.model, self.data)

    def render(self) -> np.ndarray | None:
        """Sync the passive viewer (human) or return an RGB frame (rgb_array)."""
        match self._render_mode:
            case "human":
                self._sync_viewer()
                return None
            case "rgb_array":
                return self._render_rgb()
            case _:
                return None

    def is_open(self) -> bool:
        """Return False once a human viewer window has been closed by the user."""
        return self._viewer is None or self._viewer.is_running()

    def close(self) -> None:
        """Close the viewer and offscreen renderer, if any were created."""
        try:
            if self._viewer is not None:
                self._viewer.close()
        finally:
            self._viewer = None
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = None

    def _state(self) -> RobotState:
        """Snapshot of joint positions, velocities, commands, and TCP position."""
        return RobotState(
            qpos=self.data.qpos[self._ids.qpos_adr].copy(),
            qvel=self.data.qvel[self._ids.dof_adr].copy(),
            q_cmd=self._q_cmd.copy(),
            tcp_pos=self.data.site_xpos[self._ids.tcp_site].copy(),
        )

    def _apply_torque_caps(self) -> None:
        """Limit every actuated joint to the configured servo stall torque."""
        stall = self._cfg.stall_torque_nm
        self.model.jnt_actfrcrange[self._ids.joint_ids] = (-stall, stall)
        self.model.jnt_actfrclimited[self._ids.joint_ids] = 1

    def _read_limits(self) -> JointLimits:
        """Arm joint ranges from the model."""
        arm_range = self.model.jnt_range[self._ids.joint_ids[: self._ids.n_arm]]
        return JointLimits(lower=arm_range[:, 0].copy(), upper=arm_range[:, 1].copy())

    def _restore_nominal_dynamics(self) -> None:
        """Undo any randomization from a previous episode."""
        self._set_dynamics(np.ones_like(self._nominal.kp), np.ones_like(self._nominal.kp), 0.0)

    def _randomize_dynamics(self, rng: np.random.Generator) -> None:
        """Scale gains and damping and add a payload, drawn from the configured ranges."""
        cfg = self._randomization
        n = len(self._nominal.kp)
        kp_scale = rng.uniform(*cfg.kp_scale_range, size=n)
        damping_scale = rng.uniform(*cfg.damping_scale_range, size=n)
        self._set_dynamics(kp_scale, damping_scale, float(rng.uniform(*cfg.payload_mass_range_kg)))

    def _set_dynamics(self, kp_scale: np.ndarray, damping_scale: np.ndarray, payload_kg: float) -> None:
        """Write scaled actuator gains, damping, and payload mass into the model."""
        act = self._ids.actuator_ids
        kp = self._nominal.kp * kp_scale
        self.model.actuator_gainprm[act, KP_GAIN_COL] = kp
        self.model.actuator_biasprm[act, KP_BIAS_COL] = -kp
        self.model.actuator_biasprm[act, KV_BIAS_COL] = self._nominal.kv_bias * damping_scale
        self.model.dof_damping[self._ids.dof_adr] = self._nominal.dof_damping * damping_scale
        self.model.body_mass[self._ids.payload_body] = self._nominal.payload_mass + payload_kg

    def _sync_viewer(self) -> None:
        """Launch the passive viewer on first use, then push the latest state."""
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._configure_camera(self._viewer.cam)
        if self._viewer.is_running():
            self._viewer.sync()

    def _render_rgb(self) -> np.ndarray:
        """Render an offscreen RGB frame from the configured camera."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self._render_cfg.height, self._render_cfg.width)
        camera = mujoco.MjvCamera()
        self._configure_camera(camera)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def _configure_camera(self, camera: mujoco.MjvCamera) -> None:
        """Apply the configured look-at point, distance, and angles."""
        camera.lookat[:] = self._render_cfg.lookat
        camera.distance = self._render_cfg.distance
        camera.azimuth = self._render_cfg.azimuth
        camera.elevation = self._render_cfg.elevation

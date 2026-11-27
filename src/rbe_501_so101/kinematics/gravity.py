"""Static gravity torque on the arm joints from the MuJoCo model."""

from typing import Self

import mujoco
import numpy as np

from rbe_501_so101.config import SimConfig
from rbe_501_so101.envs.model_indices import ModelIndices


class GravityTorqueModel:
    """`qfrc_bias` at a pose with zero velocity: the torque each actuator must supply to hold it.

    Runs on its own `MjData`, so callers' simulation state is never touched.
    """

    def __init__(self, model: mujoco.MjModel, ids: ModelIndices) -> None:
        self._model = model
        self._ids = ids
        self._data = mujoco.MjData(model)

    @classmethod
    def from_config(cls, sim_cfg: SimConfig) -> Self:
        """Build from a private copy of the configured (nominal) model."""
        model = mujoco.MjModel.from_xml_path(str(sim_cfg.model_path))
        return cls(model, ModelIndices.from_model(model, sim_cfg))

    def torque(self, q: np.ndarray) -> np.ndarray:
        """Arm-joint gravity torque (N*m) for a full joint vector (arm joints, then gripper)."""
        self._data.qpos[self._ids.qpos_adr] = q
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        return self._data.qfrc_bias[self._ids.arm_dof_adr].copy()

    def actuator_kp(self) -> np.ndarray:
        """Position-actuator stiffness (N*m/rad) of the arm joints in this model."""
        return self._model.actuator_gainprm[self._ids.actuator_ids[: self._ids.n_arm], 0].copy()

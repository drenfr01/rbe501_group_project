from pathlib import Path
import time

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "so101"
    / "so101.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("Bodies:", model.nbody)
print("Joints:", model.njnt)
print("Actuators:", model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Nice default camera view
    viewer.cam.lookat[:] = [0.02, 0.00, 0.48]
    viewer.cam.distance = 1.05
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -22

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
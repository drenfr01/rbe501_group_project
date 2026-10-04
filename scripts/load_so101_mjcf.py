from pathlib import Path
import time
import math

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "so101"
    / "so101.xml"
)

print("Loading MJCF model:")
print(MODEL_PATH)
print()

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Could not find MJCF model:\n{MODEL_PATH}"
    )

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Keep gravity off for this kinematic comparison
model.opt.gravity[:] = 0

print("MJCF SO-101 loaded successfully!")
print()

print("MODEL INFORMATION")
print("-----------------")
print("Number of bodies:", model.nbody)
print("Number of joints:", model.njnt)
print("Number of DOFs:", model.nv)
print("Number of actuators:", model.nu)
print()

print("JOINTS")
print("------")

for i in range(model.njnt):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )
    print(f"{i}: {name}")


def set_joint_angle(joint_name, angle_degrees):

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    if joint_id == -1:
        raise ValueError(f"Joint '{joint_name}' not found.")

    qpos_index = model.jnt_qposadr[joint_id]

    data.qpos[qpos_index] = math.radians(angle_degrees)


# Same pose we already tested with the URDF
set_joint_angle("shoulder_pan", 20)
set_joint_angle("shoulder_lift", -20)
set_joint_angle("elbow_flex", 40)
set_joint_angle("wrist_flex", 15)
set_joint_angle("wrist_roll", 30)
set_joint_angle("gripper", 10)

mujoco.mj_forward(model, data)

print()
print("Opening MuJoCo viewer...")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        mujoco.mj_forward(model, data)
        viewer.sync()

        time.sleep(0.01)
from pathlib import Path
import time
import math

import mujoco
import mujoco.viewer


# ---------------------------------------------------------
# Find the project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "models"
    / "so101"
    / "so101_new_calib.urdf"
)


# ---------------------------------------------------------
# Make sure the file actually exists
# ---------------------------------------------------------

print("Looking for SO-101 URDF at:")
print(URDF_PATH)
print()

if not URDF_PATH.exists():
    raise FileNotFoundError(
        f"Could not find SO-101 URDF:\n{URDF_PATH}"
    )


# ---------------------------------------------------------
# Load the SO-101 into MuJoCo
# ---------------------------------------------------------

print("Loading SO-101...")

model = mujoco.MjModel.from_xml_path(str(URDF_PATH))
data = mujoco.MjData(model)

# Keep gravity off for this kinematics test
model.opt.gravity[:] = 0

print("SO-101 loaded successfully!")
print()


# ---------------------------------------------------------
# Print useful model information
# ---------------------------------------------------------

print("MODEL INFORMATION")
print("-----------------")
print("Number of bodies:", model.nbody)
print("Number of joints:", model.njnt)
print("Number of DOFs:", model.nv)
print("Number of actuators:", model.nu)
print()


# ---------------------------------------------------------
# Print joint names
# ---------------------------------------------------------

print("JOINTS")
print("------")

for i in range(model.njnt):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    print(f"{i}: {name}")


# ---------------------------------------------------------
# Helper function: set joint angle by name
# ---------------------------------------------------------

def set_joint_angle(joint_name, angle_degrees):
    """
    Set a MuJoCo joint position using degrees.
    """

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    if joint_id == -1:
        raise ValueError(f"Joint '{joint_name}' was not found.")

    qpos_index = model.jnt_qposadr[joint_id]

    angle_radians = math.radians(angle_degrees)

    data.qpos[qpos_index] = angle_radians

    print(
        f"{joint_name:15s} = "
        f"{angle_degrees:6.1f} deg "
        f"({angle_radians:.3f} rad)"
    )


# ---------------------------------------------------------
# Set a test pose
# ---------------------------------------------------------

print()
print("SETTING TEST POSE")
print("-----------------")

set_joint_angle("shoulder_pan", 20)
set_joint_angle("shoulder_lift", -20)
set_joint_angle("elbow_flex", 40)
set_joint_angle("wrist_flex", 15)
set_joint_angle("wrist_roll", 30)
set_joint_angle("gripper", 10)


# Recalculate the entire robot pose
mujoco.mj_forward(model, data)


# ---------------------------------------------------------
# Open passive viewer
# ---------------------------------------------------------

print()
print("Opening MuJoCo viewer...")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # Keep recomputing geometry without advancing physics
        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(0.01)
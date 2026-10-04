from pathlib import Path
import time
import math

import mujoco
import mujoco.viewer


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "so101"
    / "so101.xml"
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Loading:")
print(MODEL_PATH)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)


# ---------------------------------------------------------
# TEMPORARY DEBUG SETTINGS
# ---------------------------------------------------------

# Disable gravity for this actuator test
model.opt.gravity[:] = 0

# Disable ALL contact/collision forces temporarily
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

print()
print("MODEL INFORMATION")
print("-----------------")
print("Joints:", model.njnt)
print("Actuators:", model.nu)
print()


# ---------------------------------------------------------
# Find shoulder_pan actuator
# ---------------------------------------------------------

actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "shoulder_pan_motor"
)

if actuator_id == -1:
    raise ValueError(
        "Could not find actuator 'shoulder_pan_motor'."
    )

print("shoulder_pan actuator ID:", actuator_id)


# ---------------------------------------------------------
# Find shoulder_pan joint
# ---------------------------------------------------------

joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "shoulder_pan"
)

if joint_id == -1:
    raise ValueError(
        "Could not find joint 'shoulder_pan'."
    )

qpos_index = model.jnt_qposadr[joint_id]


# ---------------------------------------------------------
# Find other joints that we will TEMPORARILY freeze
# ---------------------------------------------------------

locked_joint_names = [
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

locked_qpos_indices = []

for name in locked_joint_names:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    if jid == -1:
        raise ValueError(
            f"Could not find joint '{name}'."
        )

    qpos_addr = model.jnt_qposadr[jid]

    locked_qpos_indices.append(qpos_addr)


# ---------------------------------------------------------
# Reset simulation
# ---------------------------------------------------------

mujoco.mj_resetData(model, data)

# Start shoulder_pan at 0 degrees
data.qpos[qpos_index] = 0.0

# Start all other joints at 0 degrees
for index in locked_qpos_indices:
    data.qpos[index] = 0.0

mujoco.mj_forward(model, data)


# ---------------------------------------------------------
# Set shoulder target
# ---------------------------------------------------------

target_degrees = 30.0
target_radians = math.radians(target_degrees)

data.ctrl[actuator_id] = target_radians

print()
print(f"Target shoulder_pan: {target_degrees:.1f} degrees")
print()


# ---------------------------------------------------------
# Run simulation
# ---------------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:

    start_time = time.time()

    last_print_second = -1

    while viewer.is_running():

        # -------------------------------------------------
        # Advance physics
        # -------------------------------------------------

        mujoco.mj_step(model, data)


        # -------------------------------------------------
        # TEMPORARY:
        # Force all other joints to remain at zero
        # -------------------------------------------------

        for index in locked_qpos_indices:
            data.qpos[index] = 0.0
            data.qvel[index] = 0.0


        # Recompute kinematics after forcing those joints
        mujoco.mj_forward(model, data)


        # -------------------------------------------------
        # Print shoulder angle once per second
        # -------------------------------------------------

        elapsed = time.time() - start_time

        current_second = int(elapsed)

        if current_second != last_print_second:

            last_print_second = current_second

            current_angle_degrees = math.degrees(
                data.qpos[qpos_index]
            )

            print(
                f"Target: {target_degrees:6.2f} deg | "
                f"Actual: {current_angle_degrees:6.2f} deg | "
                f"Control: {data.ctrl[actuator_id]:.3f} rad"
            )


        # -------------------------------------------------
        # Update viewer
        # -------------------------------------------------

        viewer.sync()

        # Run approximately in real time
        time.sleep(model.opt.timestep)
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
# Temporary debug settings
# ---------------------------------------------------------

# Keep gravity off for now
#model.opt.gravity[:] = 0

# Keep collisions off for now
#model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT


# ---------------------------------------------------------
# Print model info
# ---------------------------------------------------------

print()
print("MODEL INFORMATION")
print("-----------------")
print("Joints:", model.njnt)
print("Actuators:", model.nu)
print()


# ---------------------------------------------------------
# Desired joint targets in degrees
# ---------------------------------------------------------

targets_deg = {
    "shoulder_pan_motor": 20.0,
    "shoulder_lift_motor": -20.0,
    "elbow_flex_motor": 40.0,
    "wrist_flex_motor": 15.0,
    "wrist_roll_motor": 30.0,
    "gripper_motor": 10.0,
}


# ---------------------------------------------------------
# Apply target positions to actuators
# ---------------------------------------------------------

for actuator_name, target_deg in targets_deg.items():

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name
    )

    if actuator_id == -1:
        raise ValueError(
            f"Could not find actuator '{actuator_name}'."
        )

    target_rad = math.radians(target_deg)

    data.ctrl[actuator_id] = target_rad

    print(
        f"{actuator_name:22s} -> "
        f"{target_deg:6.1f} deg "
        f"({target_rad:.3f} rad)"
    )


# ---------------------------------------------------------
# Reset initial state
# ---------------------------------------------------------

mujoco.mj_forward(model, data)


# ---------------------------------------------------------
# Helper to read a joint angle in degrees
# ---------------------------------------------------------

def get_joint_angle_deg(joint_name):

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    if joint_id == -1:
        raise ValueError(
            f"Could not find joint '{joint_name}'."
        )

    qpos_index = model.jnt_qposadr[joint_id]

    return math.degrees(
        data.qpos[qpos_index]
    )


# ---------------------------------------------------------
# Map actuator names to joint names
# ---------------------------------------------------------

actuator_to_joint = {
    "shoulder_pan_motor": "shoulder_pan",
    "shoulder_lift_motor": "shoulder_lift",
    "elbow_flex_motor": "elbow_flex",
    "wrist_flex_motor": "wrist_flex",
    "wrist_roll_motor": "wrist_roll",
    "gripper_motor": "gripper",
}


# ---------------------------------------------------------
# Run simulation
# ---------------------------------------------------------

print()
print("Running all 6 actuators...")
print()

with mujoco.viewer.launch_passive(model, data) as viewer:

    start_time = time.time()

    last_print_second = -1

    while viewer.is_running():

        # Advance the physics
        mujoco.mj_step(model, data)

        elapsed = time.time() - start_time
        current_second = int(elapsed)

        # Print once per second
        if current_second != last_print_second:

            last_print_second = current_second

            print(f"\nTime: {elapsed:.1f} s")

            for actuator_name, joint_name in actuator_to_joint.items():

                actual_deg = get_joint_angle_deg(joint_name)
                target_deg = targets_deg[actuator_name]

                print(
                    f"{joint_name:15s} | "
                    f"Target: {target_deg:6.1f} deg | "
                    f"Actual: {actual_deg:6.2f} deg"
                )

        viewer.sync()

        # Approximate real-time simulation
        time.sleep(model.opt.timestep)
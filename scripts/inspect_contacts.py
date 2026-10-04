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
# Desired actuator targets
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
# Apply target positions
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

    data.ctrl[actuator_id] = math.radians(target_deg)


# ---------------------------------------------------------
# Helper function:
# get body / geom / mesh information
# ---------------------------------------------------------

def get_geom_info(geom_id):

    geom_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id
    )

    body_id = model.geom_bodyid[geom_id]

    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id
    )

    # If this geom is a mesh, geom_dataid points to the mesh ID.
    # For non-mesh geoms it may point to some other asset type,
    # so we check the geom type first.
    mesh_name = None
    mesh_id = -1

    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:

        mesh_id = model.geom_dataid[geom_id]

        if mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_MESH,
                mesh_id
            )

    return {
        "geom_id": geom_id,
        "geom_name": geom_name,
        "body_id": body_id,
        "body_name": body_name,
        "mesh_id": mesh_id,
        "mesh_name": mesh_name,
    }


# ---------------------------------------------------------
# Run simulation and inspect contacts
# ---------------------------------------------------------

print()
print("Inspecting contacts...")
print()

with mujoco.viewer.launch_passive(model, data) as viewer:

    last_print_second = -1

    while viewer.is_running():

        # Advance physics normally
        mujoco.mj_step(model, data)

        current_second = int(data.time)

        # Print once per second
        if current_second != last_print_second:

            last_print_second = current_second

            print()
            print(f"TIME: {data.time:.2f} s")
            print(f"Number of contacts: {data.ncon}")
            print("----------------------------------------")

            seen_pairs = set()

            for i in range(data.ncon):

                contact = data.contact[i]

                info1 = get_geom_info(contact.geom1)
                info2 = get_geom_info(contact.geom2)

                # Avoid printing the same geom pair repeatedly
                pair = tuple(
                    sorted([
                        info1["geom_id"],
                        info2["geom_id"]
                    ])
                )

                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)

                print(
                    f"Geom {info1['geom_id']} "
                    f"| body={info1['body_name']} "
                    f"| geom_name={info1['geom_name']} "
                    f"| mesh={info1['mesh_name']}"
                )

                print("    <-->")

                print(
                    f"Geom {info2['geom_id']} "
                    f"| body={info2['body_name']} "
                    f"| geom_name={info2['geom_name']} "
                    f"| mesh={info2['mesh_name']}"
                )

                print()

        viewer.sync()

        time.sleep(model.opt.timestep)
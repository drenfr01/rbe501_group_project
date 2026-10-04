from pathlib import Path
import math

import numpy as np
import mujoco


# =========================================================
# LOAD MODEL
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "so101"
    / "so101.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)


# =========================================================
# PUT ROBOT IN A KNOWN CONFIGURATION
# =========================================================

joint_targets_deg = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 40.0,
    "wrist_flex": 15.0,
    "wrist_roll": 0.0,

    # Measure the fingertips with gripper open
    "gripper": 40.0,
}

for joint_name, angle_deg in joint_targets_deg.items():

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

    data.qpos[qpos_index] = math.radians(angle_deg)


mujoco.mj_forward(model, data)


# =========================================================
# FIND RELEVANT BODIES
# =========================================================

gripper_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "gripper_link"
)

moving_jaw_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "moving_jaw_so101_v1_link"
)

if gripper_body_id == -1:
    raise ValueError("Could not find gripper_link.")

if moving_jaw_body_id == -1:
    raise ValueError(
        "Could not find moving_jaw_so101_v1_link."
    )


# =========================================================
# FIND THE TWO JAW MESHES
# =========================================================

fixed_mesh_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_MESH,
    "wrist_roll_follower_so101_v1"
)

moving_mesh_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_MESH,
    "moving_jaw_so101_v1"
)

if fixed_mesh_id == -1:
    raise ValueError(
        "Could not find fixed jaw mesh."
    )

if moving_mesh_id == -1:
    raise ValueError(
        "Could not find moving jaw mesh."
    )


# =========================================================
# FIND WHICH GEOM USES EACH MESH
# =========================================================

def find_geom_using_mesh(mesh_id, preferred_body_id=None):

    candidates = []

    for geom_id in range(model.ngeom):

        if (
            model.geom_type[geom_id]
            == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom_dataid[geom_id] == mesh_id
        ):

            candidates.append(geom_id)

            if (
                preferred_body_id is not None
                and model.geom_bodyid[geom_id]
                == preferred_body_id
            ):
                return geom_id

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise ValueError(
            f"No geom found using mesh ID {mesh_id}"
        )

    raise ValueError(
        f"Multiple geoms use mesh ID {mesh_id}. "
        "Could not uniquely identify the correct one."
    )


fixed_geom_id = find_geom_using_mesh(
    fixed_mesh_id,
    gripper_body_id
)

moving_geom_id = find_geom_using_mesh(
    moving_mesh_id,
    moving_jaw_body_id
)


# =========================================================
# CONVERT MESH VERTICES TO WORLD COORDINATES
# =========================================================

def get_world_vertices(mesh_id, geom_id):

    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]

    vertices_local = model.mesh_vert[
        start:start + count
    ].copy()

    geom_position = data.geom_xpos[
        geom_id
    ].copy()

    geom_rotation = data.geom_xmat[
        geom_id
    ].reshape(3, 3)

    vertices_world = (
        vertices_local @ geom_rotation.T
        + geom_position
    )

    return vertices_world


fixed_vertices_world = get_world_vertices(
    fixed_mesh_id,
    fixed_geom_id
)

moving_vertices_world = get_world_vertices(
    moving_mesh_id,
    moving_geom_id
)


# =========================================================
# DETERMINE THE GRIPPER'S FORWARD DIRECTION
# =========================================================
#
# The official end-effector frame extends roughly along
# negative local Z of gripper_link.
# =========================================================

gripper_rotation = data.xmat[
    gripper_body_id
].reshape(3, 3)

forward_local = np.array(
    [0.0, 0.0, -1.0]
)

forward_world = (
    gripper_rotation @ forward_local
)

forward_world /= np.linalg.norm(
    forward_world
)


# =========================================================
# ESTIMATE FINGERTIP CENTER FROM EACH MESH
# =========================================================
#
# Find the furthest vertices in the forward direction,
# then average vertices lying within 2 mm of that extreme.
# =========================================================

def estimate_tip(vertices_world, forward_direction):

    projections = (
        vertices_world @ forward_direction
    )

    maximum_projection = np.max(
        projections
    )

    threshold = 0.002  # 2 mm

    tip_vertices = vertices_world[
        projections
        >= maximum_projection - threshold
    ]

    tip_world = np.mean(
        tip_vertices,
        axis=0
    )

    return tip_world


fixed_tip_world = estimate_tip(
    fixed_vertices_world,
    forward_world
)

moving_tip_world = estimate_tip(
    moving_vertices_world,
    forward_world
)


# =========================================================
# WORLD -> LOCAL BODY COORDINATES
# =========================================================

def world_to_body_local(
    world_point,
    body_id
):

    body_position = data.xpos[
        body_id
    ].copy()

    body_rotation = data.xmat[
        body_id
    ].reshape(3, 3)

    local_point = (
        body_rotation.T
        @ (
            world_point
            - body_position
        )
    )

    return local_point


# Fixed jaw site belongs to gripper_link
fixed_tip_local = world_to_body_local(
    fixed_tip_world,
    gripper_body_id
)

# Moving jaw site belongs to moving_jaw body
moving_tip_local = world_to_body_local(
    moving_tip_world,
    moving_jaw_body_id
)


# =========================================================
# DYNAMIC MIDPOINT FOR THIS 40 DEG CONFIGURATION
# =========================================================

grasp_center_world = (
    fixed_tip_world
    + moving_tip_world
) / 2.0

jaw_distance = np.linalg.norm(
    fixed_tip_world
    - moving_tip_world
)


# =========================================================
# PRINT RESULTS
# =========================================================

print()
print("=======================================")
print("SO-101 FINGERTIP SITE CALCULATION")
print("=======================================")

print()
print("Fixed fingertip [world]:")
print(fixed_tip_world)

print()
print("Fixed fingertip [gripper_link local]:")
print(fixed_tip_local)

print()
print("Moving fingertip [world]:")
print(moving_tip_world)

print()
print(
    "Moving fingertip "
    "[moving_jaw_so101_v1_link local]:"
)
print(moving_tip_local)

print()
print(
    f"Jaw-tip separation: "
    f"{jaw_distance * 1000:.2f} mm"
)

print()
print("Dynamic midpoint [world]:")
print(grasp_center_world)


# =========================================================
# OUTPUT EXACT MJCF SITES
# =========================================================

print()
print("=======================================")
print("COPY THESE INTO SO101.XML")
print("=======================================")

print()
print("Inside <body name=\"gripper_link\">:")
print()

print(
    '<site '
    'name="fixed_tip_site" '
    f'pos="{fixed_tip_local[0]:.6f} '
    f'{fixed_tip_local[1]:.6f} '
    f'{fixed_tip_local[2]:.6f}" '
    'size="0.006" '
    'rgba="1 0 0 1"/>'
)

print()
print(
    'Inside '
    '<body name="moving_jaw_so101_v1_link">:'
)
print()

print(
    '<site '
    'name="moving_tip_site" '
    f'pos="{moving_tip_local[0]:.6f} '
    f'{moving_tip_local[1]:.6f} '
    f'{moving_tip_local[2]:.6f}" '
    'size="0.006" '
    'rgba="0 1 0 1"/>'
)
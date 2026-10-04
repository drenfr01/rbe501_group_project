from pathlib import Path
import time
import math

import numpy as np
import mujoco
import mujoco.viewer


# =========================================================
# PROJECT / MODEL
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
# TEMPORARY DIAGNOSTIC
# =========================================================
#
# Set this to True temporarily to test whether the
# remaining end-effector error comes from gravity.
#
# After the diagnostic, set it back to False.
#
# =========================================================

DISABLE_GRAVITY_FOR_TEST = True

if DISABLE_GRAVITY_FOR_TEST:
    model.opt.gravity[:] = 0


# =========================================================
# JOINTS USED BY IK
# =========================================================

ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


# =========================================================
# FIND BOTH FINGERTIP SITES
# =========================================================

fixed_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "fixed_tip_site"
)

moving_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "moving_tip_site"
)

if fixed_site_id == -1:
    raise ValueError(
        "Could not find site 'fixed_tip_site'."
    )

if moving_site_id == -1:
    raise ValueError(
        "Could not find site 'moving_tip_site'."
    )


# =========================================================
# FIND CUBE
# =========================================================

cube_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "cube"
)

if cube_body_id == -1:
    raise ValueError(
        "Could not find body 'cube'."
    )


# =========================================================
# BUILD JOINT INDEX LISTS
# =========================================================

joint_ids = []
qpos_indices = []
dof_indices = []

for joint_name in ARM_JOINTS:

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    if joint_id == -1:
        raise ValueError(
            f"Could not find joint '{joint_name}'."
        )

    joint_ids.append(joint_id)

    qpos_indices.append(
        model.jnt_qposadr[joint_id]
    )

    dof_indices.append(
        model.jnt_dofadr[joint_id]
    )


# =========================================================
# INITIAL GUESS
# =========================================================

initial_guess_deg = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 40.0,
    "wrist_flex": 15.0,
    "wrist_roll": 0.0,
}

for joint_name, qpos_index in zip(
    ARM_JOINTS,
    qpos_indices
):

    data.qpos[qpos_index] = math.radians(
        initial_guess_deg[joint_name]
    )


# =========================================================
# KEEP GRIPPER OPEN DURING IK
# =========================================================

gripper_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "gripper"
)

if gripper_joint_id == -1:
    raise ValueError(
        "Could not find joint 'gripper'."
    )

gripper_qpos_index = model.jnt_qposadr[
    gripper_joint_id
]

# 40 degrees open
data.qpos[
    gripper_qpos_index
] = math.radians(40)

mujoco.mj_forward(model, data)


# =========================================================
# GET CUBE POSITION
# =========================================================

cube_position = data.xpos[
    cube_body_id
].copy()

print()
print("Cube position:")
print(cube_position)


# =========================================================
# DEFINE IK TARGET
#
# First target = 6 cm above cube
# =========================================================

target_position = cube_position.copy()

target_position[2] += 0.06

print()
print("IK target position:")
print(target_position)


# =========================================================
# IK PARAMETERS
# =========================================================

max_iterations = 500

tolerance = 0.001       # 1 mm

damping = 0.05

step_scale = 0.5

max_joint_step = math.radians(5)


# =========================================================
# IK LOOP
# =========================================================

print()
print("Starting numerical IK...")
print()

for iteration in range(max_iterations):

    # -----------------------------------------------------
    # Update kinematics
    # -----------------------------------------------------

    mujoco.mj_forward(
        model,
        data
    )


    # -----------------------------------------------------
    # Dynamic grasp center:
    # midpoint between fixed and moving fingertips
    # -----------------------------------------------------

    fixed_position = data.site_xpos[
        fixed_site_id
    ].copy()

    moving_position = data.site_xpos[
        moving_site_id
    ].copy()

    current_position = (
        fixed_position
        + moving_position
    ) / 2.0


    # -----------------------------------------------------
    # Position error
    # -----------------------------------------------------

    error = (
        target_position
        - current_position
    )

    error_norm = np.linalg.norm(
        error
    )


    # -----------------------------------------------------
    # Check convergence
    # -----------------------------------------------------

    if error_norm < tolerance:

        print(
            f"IK converged after "
            f"{iteration} iterations."
        )

        break


    # -----------------------------------------------------
    # Fixed fingertip Jacobian
    # -----------------------------------------------------

    jac_fixed = np.zeros(
        (3, model.nv)
    )

    jac_fixed_rot = np.zeros(
        (3, model.nv)
    )

    mujoco.mj_jacSite(
        model,
        data,
        jac_fixed,
        jac_fixed_rot,
        fixed_site_id
    )


    # -----------------------------------------------------
    # Moving fingertip Jacobian
    # -----------------------------------------------------

    jac_moving = np.zeros(
        (3, model.nv)
    )

    jac_moving_rot = np.zeros(
        (3, model.nv)
    )

    mujoco.mj_jacSite(
        model,
        data,
        jac_moving,
        jac_moving_rot,
        moving_site_id
    )


    # -----------------------------------------------------
    # Keep only the 5 arm joint columns
    # -----------------------------------------------------

    J_fixed = jac_fixed[
        :,
        dof_indices
    ]

    J_moving = jac_moving[
        :,
        dof_indices
    ]


    # -----------------------------------------------------
    # Jacobian of the dynamic midpoint
    #
    # p_center = (p_fixed + p_moving) / 2
    #
    # therefore:
    #
    # J_center = (J_fixed + J_moving) / 2
    # -----------------------------------------------------

    J = (
        J_fixed
        + J_moving
    ) / 2.0


    # -----------------------------------------------------
    # Damped Least Squares
    #
    # dq =
    # J^T (J J^T + lambda^2 I)^-1 error
    # -----------------------------------------------------

    JJt = (
        J
        @ J.T
    )

    damping_matrix = (
        damping ** 2
        * np.eye(3)
    )

    dq = (
        J.T
        @ np.linalg.solve(
            JJt
            + damping_matrix,
            error
        )
    )


    # -----------------------------------------------------
    # Limit update size
    # -----------------------------------------------------

    dq *= step_scale

    dq = np.clip(
        dq,
        -max_joint_step,
        max_joint_step
    )


    # -----------------------------------------------------
    # Update joints
    # -----------------------------------------------------

    for i, (
        joint_id,
        qpos_index
    ) in enumerate(
        zip(
            joint_ids,
            qpos_indices
        )
    ):

        new_q = (
            data.qpos[
                qpos_index
            ]
            + dq[i]
        )

        joint_min = model.jnt_range[
            joint_id,
            0
        ]

        joint_max = model.jnt_range[
            joint_id,
            1
        ]

        new_q = np.clip(
            new_q,
            joint_min,
            joint_max
        )

        data.qpos[
            qpos_index
        ] = new_q


else:

    print(
        "WARNING: IK reached maximum iterations "
        "without meeting tolerance."
    )


# =========================================================
# FINAL FK
# =========================================================

mujoco.mj_forward(
    model,
    data
)

fixed_final = data.site_xpos[
    fixed_site_id
].copy()

moving_final = data.site_xpos[
    moving_site_id
].copy()

final_position = (
    fixed_final
    + moving_final
) / 2.0

final_error = (
    target_position
    - final_position
)


print()
print("=======================================")
print("IK RESULT")
print("=======================================")

print()
print("Target position:")
print(target_position)

print()
print("Reached grasp-center position:")
print(final_position)

print()
print(
    "Final position error:",
    np.linalg.norm(
        final_error
    ),
    "m"
)


# =========================================================
# SAVE SOLVED JOINT ANGLES
# =========================================================

solution = {}

print()
print("Solved joint angles:")
print("--------------------")

for joint_name, qpos_index in zip(
    ARM_JOINTS,
    qpos_indices
):

    angle = data.qpos[
        qpos_index
    ]

    solution[
        joint_name
    ] = angle

    print(
        f"{joint_name:15s}: "
        f"{math.degrees(angle):8.2f} deg"
    )


# =========================================================
# RESET ROBOT TO INITIAL POSE
# =========================================================

for joint_name, qpos_index in zip(
    ARM_JOINTS,
    qpos_indices
):

    data.qpos[
        qpos_index
    ] = math.radians(
        initial_guess_deg[
            joint_name
        ]
    )


# Keep gripper open
data.qpos[
    gripper_qpos_index
] = math.radians(40)

# Zero all velocities
data.qvel[:] = 0

mujoco.mj_forward(
    model,
    data
)


# =========================================================
# SEND IK SOLUTION TO ACTUATORS
# =========================================================

for joint_name in ARM_JOINTS:

    actuator_name = (
        joint_name
        + "_motor"
    )

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name
    )

    if actuator_id == -1:
        raise ValueError(
            f"Could not find actuator "
            f"'{actuator_name}'."
        )

    data.ctrl[
        actuator_id
    ] = solution[
        joint_name
    ]


# =========================================================
# KEEP GRIPPER OPEN
# =========================================================

gripper_actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_motor"
)

if gripper_actuator_id == -1:
    raise ValueError(
        "Could not find actuator 'gripper_motor'."
    )

data.ctrl[
    gripper_actuator_id
] = math.radians(40)


# =========================================================
# RUN PHYSICS
# =========================================================

print()
print(
    "Moving simulated robot to IK solution..."
)

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    # -----------------------------------------------------
    # Camera
    # -----------------------------------------------------

    viewer.cam.lookat[:] = [
        0.02,
        0.00,
        0.50
    ]

    viewer.cam.distance = 1.05
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -22


    last_print_second = -1


    # =====================================================
    # VIEWER LOOP
    # =====================================================

    while viewer.is_running():

        # -------------------------------------------------
        # Advance physics
        # -------------------------------------------------

        mujoco.mj_step(
            model,
            data
        )


        # -------------------------------------------------
        # Recalculate actual dynamic grasp center
        # EVERY FRAME
        # -------------------------------------------------

        fixed_pos = data.site_xpos[
            fixed_site_id
        ].copy()

        moving_pos = data.site_xpos[
            moving_site_id
        ].copy()

        grasp_center = (
            fixed_pos
            + moving_pos
        ) / 2.0


        # -------------------------------------------------
        # Distance from dynamic midpoint to target
        # -------------------------------------------------

        distance = np.linalg.norm(
            target_position
            - grasp_center
        )


        # -------------------------------------------------
        # CLEAR USER VISUALIZATION GEOMS
        # -------------------------------------------------

        viewer.user_scn.ngeom = 0


        # -------------------------------------------------
        # BLUE SPHERE:
        # actual dynamic grasp center
        # -------------------------------------------------

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array(
                [0.008, 0.008, 0.008]
            ),
            grasp_center,
            np.eye(3).flatten(),
            np.array(
                [0.0, 0.4, 1.0, 1.0]
            )
        )

        viewer.user_scn.ngeom += 1


        # -------------------------------------------------
        # MAGENTA SPHERE:
        # desired IK target
        # -------------------------------------------------

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array(
                [0.007, 0.007, 0.007]
            ),
            target_position,
            np.eye(3).flatten(),
            np.array(
                [1.0, 0.0, 1.0, 1.0]
            )
        )

        viewer.user_scn.ngeom += 1


        # -------------------------------------------------
        # Print once per simulation second
        # -------------------------------------------------

        current_second = int(
            data.time
        )

        if (
            current_second
            != last_print_second
        ):

            last_print_second = (
                current_second
            )

            print(
                f"Grasp-center distance "
                f"to target: "
                f"{distance * 1000:.2f} mm"
            )


        # -------------------------------------------------
        # Update viewer
        # -------------------------------------------------

        viewer.sync()

        time.sleep(
            model.opt.timestep
        )
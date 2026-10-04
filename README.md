# rbe501_group_project

A Gripping Story: Sim2Real with the SO-101 robot

The working MuJoCo scene is `models/so101/so101.xml`. It contains the robot,
table, floor, lighting, and a 50 mm cube. Simulation uses a 1 ms timestep,
`implicitfast`, and critically damped position actuators. Cube collisions are
enabled; the three problematic base meshes remain non-colliding.

## End-effector and IK

`gripper_site` is the TCP reference. Its position
`-0.0079 -0.000218121 -0.0981274` and quaternion `0 0 1 0` are unchanged.
The position matches `gripper_frame_joint` in the supplied SO-101 URDF.
This frame belongs to the stationary gripper body; opening the moving finger
does not move the TCP relative to that body.

Position-only damped least-squares IK uses five arm joints, their joint limits,
and one site Jacobian. Gripper opening is commanded separately and held at 40
degrees during the IK test. The test target remains 6 cm above the cube's initial
center; orientation, grasp planning, and RL are future work.

Robot-only body gravity compensation is enabled in the MJCF. This is idealized
simulation compensation, not a hardware torque controller. The cube retains
normal gravity. Actuator gains were not changed during cleanup.

## Test Scripts

Run commands from the project folder after `uv sync`. Each script supports
`--help` and starts with instructions and expected results.

| Script | Purpose | Typical command | Viewer | Normally change |
|---|---|---|---|---|
| `test_table_cube.py` | Scene/mounting sanity check with physics | `uv run python scripts/test_table_cube.py` | Yes; `--headless` disables | Scene placement in MJCF; initial pose in helper |
| `test_all_actuators.py` | Direct joint-angle commands | `uv run python scripts/test_all_actuators.py --shoulder-pan 20` | Yes; `--headless` disables | Joint CLI targets in degrees |
| `test_ik.py` | Position IK for `gripper_site` | `uv run python scripts/test_ik.py --x 0.15 --y 0 --z 0.55` | Yes; `--headless` disables | All XYZ coordinates in world meters |
| `diagnose_tracking.py` | Pure IK vs gravity/compensation tracking | `uv run python scripts/diagnose_tracking.py` | No | `--seconds` settling duration |
| `inspect_contacts.py` | Identify settled contact bodies/geoms/meshes | `uv run python scripts/inspect_contacts.py` | No | Shared actuator-test pose if needed |

Copy-paste examples:

```powershell
uv run python scripts/test_all_actuators.py
uv run python scripts/test_all_actuators.py --shoulder-pan 20 --shoulder-lift -15 --elbow-flex 40
uv run python scripts/test_all_actuators.py --gripper 30 --headless
uv run python scripts/test_ik.py
uv run python scripts/test_ik.py --x 0.15 --y 0.00 --z 0.55
uv run python scripts/test_ik.py --x 0.15 --y 0.00 --z 0.55 --headless
uv run python scripts/diagnose_tracking.py --seconds 5
```

IK takes all XYZ options or none; it preserves the default target when omitted.
Actuator options accept partial overrides, retaining omitted joint defaults.
Angles are validated against MJCF limits (shown by `--help`) before simulation.
IK is position-only; valid joint limits and reachable XYZ do not imply a
collision-free trajectory.

Defaults should produce about 0.85 mm pure IK and compensated tracking error,
versus about 11.6 mm uncompensated gravity-on error. Reports list joint errors,
contact pairs, and warnings. Cube/table contacts are expected. The gravity-off
comparison keeps the cube at its initial height; all cases use the same fixed
TCP target for a fair comparison. Short diagnostic durations may not settle.

`scripts/so101_runtime.py` provides shared loading, IK, physics, and viewer
helpers. Internal joint units are radians; CLI joint commands use degrees;
Cartesian coordinates use meters. Viewers continue physics after settling.

## Source utilities and removed experiments

`load_so101.py` is a legacy URDF inspection utility. `convert_so101_to_mjcf.py`
recreates the raw converted model and **overwrites the working MJCF**; it is not
part of the baseline test workflow. Keep it for provenance, and use a copy of
the project if reconverting.

The midpoint generator and two grasp experiments were removed because they
used a custom fingertip reference that is no longer the TCP. The standalone
shoulder test, falling-cube test, and MJCF comparison viewer were consolidated
into the scene and six-actuator tests. Their previous versions remain in Git
history. No grasp/RL strategy is implemented by this cleanup.

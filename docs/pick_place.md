# Full-task pick-and-place environment

`SO101PickPlace-v1` is a separate Gymnasium environment for the complete sequence:
reach, grasp, lift, carry, lower, release, retreat and verify resting placement.
The existing reaching and pick/lift environments and saved policies remain available.

The validated scripted controller supplies the baseline. The policy has three
normalized Cartesian residual actions with at most **1 mm vector magnitude**,
applied only during grasped carry/lower motion. Pickup, the jaw, release
authorization and retreat remain protected. On table support, vertical residual
authority is zero. A rejected residual IK/clearance proposal falls back to the
script after restoring its reference/bias state; actual contact/grasp failures
terminate the episode. Zero residual actions reproduce the validated script.

This is residual reinforcement learning for a complete task. It does not teach
all grasp/release stages from scratch, and it uses privileged simulated cube
pose, velocity and contact information.

## Use the environment

```python
import gymnasium as gym
import numpy as np
from rbe_501_so101.envs import PICK_PLACE_ENV_ID

env = gym.make(PICK_PLACE_ENV_ID)
obs, info = env.reset(seed=93000, options={"yaw_offset_deg": 30.0})
while True:
    obs, reward, terminated, truncated, info = env.step(np.zeros(3))
    if terminated or truncated:
        print(info["is_success"], info["placement_error_m"])
        break
env.close()
```

Reset samples the original reachable placement band, mass 5–20 g and one of
-30/0/+30-degree yaw offsets. The default destination is the validated nearby
12-degree same-radius table move. `options={"destination_xy": [x, y]}` specifies
an explicit resting destination within the same radial/azimuth workspace; it
appears in observations and success evaluation. Workspace validation does not
guarantee every requested transfer path succeeds.

The flat **72-value observation** has named slices in
`PickPlaceObservationLayout`: measured joints, joint velocities and command errors;
pinch/cube position; cube quaternion and linear/angular velocity; destination and
destination error; previous requested action; phase/progress/contact/mass; and
reference, bias, axis and milestone state. Three-value actions and 72-value
normalization statistics are incompatible with old 33-value pickup policies.

The horizon is 720 control periods maximum. Success terminates only after lift,
release, measured hand retreat and the 60-period final observation window, with
the cube supported, ungrasped, within 10 mm XY and 5 mm height of resting target,
linear speed <=0.01 m/s and angular speed <=0.1 rad/s. A lift alone never succeeds.
Controller/safety failure terminates; the overall time limit truncates. Reset
is required before stepping again after either outcome.

Reward is potential-based phase distance shaping (discount 0.98), a small time
cost and applied-residual cost, plus +10 for true full-task success or -10 for
controller failure. Terminal potential is absorbing on termination; a time-limit
truncation retains its next-state potential for bootstrapping. Reward terms and
physical success are reported separately in `info`.

## Short pilot and replay

```powershell
# Fresh 10k CUDA SAC pilot and paired full-task evaluation; both directories must be new
uv run --locked python -m scripts.sim2real.analyze.pick_place_pilot --out experiments/pick_place_pilot_repeat --model-dir checkpoints/pick_place_pilot_repeat

# Replay the locally saved new full-task policy
uv run --locked python -m scripts.sim2real.analyze.pick_place_pilot --replay --viewer --model-dir checkpoints/pick_place_v1_pilot --seed 93000 --mass-kg 0.01 --yaw-offset-deg 0 --out experiments/full_policy_viewer_1
```

Add `--scripted` to replay the matching zero-residual baseline. Omit `--viewer`
for a headless replay with phase images and physical qpos trace. Viewer playback
uses the recorded trajectory after the environment rollout.

New runs store the complete configuration, reward/layout/controller snapshot,
model/assets/calibration/runtime hashes, seed and training settings in
`run_config.json`. Loading requires that snapshot to match. Runtime/text hashes
normalize CRLF/LF; binary mesh assets are hashed unchanged. Share the entire run
directory with matching `model.zip` and `vecnormalize.pkl` to reproduce weights.
Checkpoint/replay-buffer binaries remain local and ignored by Git.

The full-task pilot is a short stability check against an already successful
script. Matching scripted success does not establish added RL benefit or justify
a long training run automatically. Hardware orientation/contact sensing and
gripper/contact dynamics still need physical validation.

The [frozen carry-push comparison](../reports/findings/2026-11-26_pick_place_disturbance.md)
then tests the saved full-task policy and its scripted baseline without training.
Both complete 320/320 pushed placements and 40/40 controls. It establishes no
useful added RL robustness; see the report for paired errors, recovery definition
and limits. Reproduce with `scripts.sim2real.analyze.pick_place_disturbance` and
a new `--out` directory.

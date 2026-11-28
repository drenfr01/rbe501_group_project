# Pick-and-lift with object mass as a parameter

`SO101PickLift-v0` grasps a cube from the table and lifts it. The cube's
mass `m` is the experimental parameter: train one policy on a fixed mass
and one on a random mass per episode, then compare both across masses.
The policy never observes `m`.

```bash
python scripts/run_pick_lift.py --mass-kg 0.01
python scripts/train_pick_lift.py train --mass-mode fixed --mass-kg 0.01 --seeds 0
python scripts/train_pick_lift.py train --mass-mode random --mass-range 0.005 0.02 --seeds 0
python scripts/train_pick_lift.py compare "checkpoints/pick-lift checkpoint/pick_lift_fixed_10g_25mm" "checkpoints/pick-lift checkpoint/pick_lift_random_5_20g_25mm" \
    --out logs/pick_lift_5_20g_25mm_sweep.json --plot logs/pick_lift_5_20g_25mm_sweep.png
```

`compare` runs the scripted baseline and each trained run on the same cube
placements at each mass. It reports success rate and final lift height.
The default evaluation masses are 5, 10, 15, and 20 g. Run folders include
their training masses and cube side so a new setting cannot reuse an
incompatible checkpoint. Earlier 30 mm runs remain in their original folders.

Rerun `train` with a larger `--timesteps` value to continue from the saved
checkpoint. The value is the total desired steps. Each seed saves its model,
replay buffer, and observation normalization statistics every 25,000 steps
and on interruption.

The shared 25 mm runs are in `checkpoints/pick-lift checkpoint/`. Their replay
buffers use Git LFS.

The action has six relative joint commands: five arm joints and the gripper.
The 33-value observation includes joint positions, speeds, command errors,
previous action, pinch point, cube center, and their difference. An episode
lasts 200 steps (about 6.6 seconds). Success requires both finger pads to
touch the cube while its center is at least 4 cm above its starting height.

The scripted controller hovers 5 cm above the cube, descends, closes the jaws,
and lifts 7 cm. With `--control residual` (the default), the learned policy
adds a correction to those scripted actions. With `--control direct`, it
commands the arm and gripper by itself.

The pick-and-lift task builds its own MuJoCo model from `so101.xml` at runtime.
It uses fingertip collision shapes, a 25 mm cube, and contact settings that
allow a held cube to stay between the jaws. The source model remains unchanged.
The simulated motor gains are placeholders, so these results do not measure
real SO-101 payload limits. The simulation currently assumes fixed contact
settings; verify that their friction matches the printed material.

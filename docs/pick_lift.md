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

## Calibration integration boundary

Merging the reaching calibration code does not change existing pick-and-lift
checkpoints or their default dynamics. Calibrated reaching presets construct
the measured-stiffness backend and their configured servo/friction and sensing
wrappers through `envs/factory.py`.

For a controlled pick-and-lift stiffness experiment, Python callers can set
`PickLiftConfig.sim.calibration_path` to the measured actuator JSON. The shared
backend applies those gains to the compiled grasp model before capturing its
nominal dynamics, so resets preserve them. This does **not** enable the reaching
servo-effects wrappers or calibrate the moving gripper, contact friction, or
payload response. Existing pick-and-lift CLI commands retain their original
dynamics; a fully calibrated manipulation preset needs separate validation.

See `reports/findings/2026-11-26_calibration_pick_lift_integration.md` for the
integration checks and the small original-versus-measured-stiffness comparison.

## Experimental guarded setup validation

The opt-in validation below compares a stable-contact lift guard, 0–8 mm grasp
waypoint offsets, and manipulation access to the existing arm servo/sensing
wrappers. It does not change any default controller or checkpoint. Its frozen
candidate is **not approved for retraining or hardware deployment**: held-out
tests still find table contact and grasp loss.

```powershell
uv run --locked python -m scripts.sim2real.analyze.validate_pick_lift_setup --out experiments/guarded_setup_repeat
uv run --locked python -m scripts.sim2real.analyze.replay_pick_lift --controller random_5_20g --mass-kg 0.005 --placement-seed 20008 --calibrated --guarded --viewer --out experiments/guarded_drop_viewer
```

Output directories must be new. Add `--arm-effects` to the replay to stress the
controller with arm stiction, clipping, fixed delay, and encoder quantization.
This is not a hardware-calibrated gripper/contact/perception model. The guard
uses simulation contact truth, and cannot catch a cube after it has slipped.
See `reports/findings/2026-11-26_guarded_pick_lift_validation.md`.

## Experimental motion alignment

The opt-in calibrated motion ablation and independent results are documented in
[the motion findings](../reports/findings/2026-11-26_pick_lift_motion_ablation.md).
The candidate improves scripted table clearance but misses the held-out success
gate with arm effects; default control and training presets remain unchanged.
Use replay's `--motion-variant gravity_yaw_depth4 --calibrated --guarded` flags
to inspect it, optionally adding `--arm-effects`. No retraining was performed.

The subsequent [wrist acquisition findings](../reports/findings/2026-11-26_pick_lift_wrist_acquisition.md)
document the opt-in `--motion-variant outer_wrist` candidate: 200/200 scripted
lifts with stiffness alone and 199/200 with arm effects on fresh placements,
with no sampled table contacts. It passes the observed simulation gate; training
defaults and saved policies are unchanged.

The [calibrated SAC pilot](../reports/findings/2026-11-26_calibrated_lift_pilot.md)
adds a separate serialized training preset and a 10k-transition fresh-policy test.
It failed held-out evaluation (29/80 lifts versus scripted 79/80, with table
contacts), so its weights are not promoted. The report includes the dedicated
training/replay commands and locations of the local checkpoint companions.

The subsequent [lift-only residual pilot](../reports/findings/2026-11-26_lift_only_residual_pilot.md)
protects acquisition and the jaw, bounds lift corrections and checks nominal table
clearance/holding height. Its fresh v2 policy matches the script at 78/80 lifts
with zero sampled table contacts; this passes the stability gate but does not
establish an RL improvement. Use the dedicated `limited_lift_pilot` commands in
that report. Legacy defaults and reaching policies remain unchanged.

[Acquisition and holding probes](../reports/findings/2026-11-26_acquisition_holding_checks.md)
then preserve both controllers and diagnose premature stationary-finger contact
during the failing approach. In 720 paired holding-force episodes, both retain
all 312 eligible disturbed holds; RL shows only a tiny simulated cube-motion
reduction and more TCP motion. No training or controller changes were made.

The [raised-approach experiment](../reports/findings/2026-11-26_raised_approach.md)
tests the next acquisition change separately. Raise-only improves arm-effect
held-out lifts from 195/200 to 198/200 and removes observed initial approach
contacts, but still fails the pre-close cube-motion gate. Alignment correction
variants introduce table contacts and are rejected. Defaults and weights remain
unchanged; use only the opt-in experiment/replay commands in that report.

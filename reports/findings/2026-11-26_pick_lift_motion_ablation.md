# Pick/lift motion ablation — 2026-11-26

The scripted grasp improves with cube-face alignment, bounded gravity feedforward,
and a 4 mm deeper waypoint. It does **not** pass the independent reliability gate
with the experimental arm effects. Keep this opt-in; do not promote defaults or
start retraining against it yet.

## Scope and method

Base commit: `fb98b09`, integration/calibrated-pick-lift. No training. All cases use
measured stiffness and the existing five-consecutive-contact lift guard. Compare
stiffness alone against the experimental arm stiction, torque clipping, delay and
encoder quantization adapter. These are simulation assumptions, not a validated
physical manipulation model. Gripper calibration is still missing.

Seven scripted variants, four masses (5/10/15/20 g), two dynamics modes, ten paired
placements (60000–60009): **560 tuning episodes**. Freeze the selection in
`frozen_selection.json` before evaluating either saved policy. Independent check:
baseline and selected candidate, three controllers, both modes, four masses,
twenty placements (61000–61019): **960 held-out episodes**. These are placement
seeds; the two learned checkpoints remain training seed 0.

Tuning gate: at least 95% final success and at most 5% robot/table contact episodes
in each dynamics mode, pooling four masses. Contact counts sample control-step
boundaries unless explicitly described as a physics-step audit.

## What was changed experimentally

`AlignedPickLift` subclasses the existing scripted controller. Disabled options
reproduce its actions. Existing PinchIK mathematics, physical gains, collision
parameters and source model are unchanged.

- Seed wrist roll toward the nearest cube face. This uses the current TablePlacer
  rule that cube yaw equals its azimuth around the pan axis. It is **not** general
  pose perception, and will need an orientation estimate for arbitrary real cubes.
- Command `q_ideal + gravity_torque / kp`, with correction clipped to 0.1 rad and
  joint limits. This is command feedforward, not a change to body gravcomp or gains.
- Set the experimental grasp waypoint 4 mm below cube center, retaining the
  existing pinch reference and collision geometry.

An ideal top-down pose for seed 60000 has about 6.38 mm fixed-pad/table clearance;
actual tracking sag can consume it. Oblique placements can leave the original
jaw yaw about 26 degrees from a square face. These motivate the experiments but
do not establish a general contact model.

## Tuning results

Each row reports the worse success rate and worse table-contact rate across the
two dynamics modes (40 episodes per mode).

| Variant | Worst final success | Worst table contact |
|---|---:|---:|
| Baseline | 90% | 100% |
| Gravity feedforward only | 20% | 0% |
| Face yaw only | 100% | 100% |
| Gravity + yaw | 40% | 0% |
| Gravity + yaw + 4 mm depth | 100% | 0% |
| Upright lift only | 90% | 100% |
| Gravity + yaw + depth + upright lift | 100% | 0% |

Freeze `gravity_yaw_depth4`; upright lift adds no tuning benefit. Feedforward alone
is insufficient: removing sag changes finger seating, and grasp depth matters.

## Held-out results

Each cell is final successes / robot-table-contact episodes, out of **80**.
Success requires the task's final two-pad contact and at least 4 cm lift.

| Dynamics | Controller | Baseline | Selected motion |
|---|---|---:|---:|
| Stiffness only | Scripted | 75 / 77 | 80 / 0 |
| Stiffness only | Fixed-10g policy | 74 / 76 | 70 / 72 |
| Stiffness only | Random-mass policy | 76 / 69 | 71 / 43 |
| Arm effects | Scripted | 71 / 0 | 72 / 0 |
| Arm effects | Fixed-10g policy | 54 / 77 | 66 / 58 |
| Arm effects | Random-mass policy | 79 / 60 | 70 / 13 |

No MuJoCo warnings in any tuning or held-out episode. Candidate scripted success
is 100% with stiffness only, **90% with arm effects**, below the frozen 95% gate.
Old residual policies can reintroduce table contacts even with the guard; the
guard checks contact stability before lift, not motion clearance.

The eight scripted candidate failures with arm effects are seeds 61010 and 61014
at all four masses. At 10 g, 61010 never acquires both pads; 61014 acquires them
for one control step (86) and loses them at 87. Neither releases the lift gate.
This points to acquisition/seating, not a failure after an executed lift.

## Additional checks and evidence

Physics-step audit: baseline and candidate, both dynamics modes, 10 g, seeds
61000/61010/61014: **12 episodes**, 6,600 physics steps each. The six candidate
episodes have zero robot/table-contact substeps; stiffness-only baseline cases
have 480/1605/1699. This limited audit does not prove all 960 episodes have no
transient contacts. Audit outcomes match the corresponding main evaluation.

Artifacts: `experiments/2026-11-26_pick_lift_motion_ablation/` contains tuning,
frozen selection, held-out episodes, source/version manifest, physics-step audit,
and full traces plus phase PNGs for a success and both 10 g failures. Input SHA256
checks confirm the model, calibration file, saved policies, normalizers and saved
run configurations are unchanged.

Tests: **375 passed, one optional LeRobot test skipped** using
`.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider`. New motion and replay
tests were first observed failing before implementation and then passed.

## Replay and next decision

From the project root, choose a new output directory each run:

```powershell
uv run --locked python -m scripts.sim2real.analyze.replay_pick_lift --controller scripted --mass-kg 0.01 --placement-seed 61010 --calibrated --guarded --arm-effects --motion-variant gravity_yaw_depth4 --viewer --out experiments/motion_failure_viewer
```

Use seed 61000 for the successful case or 61014 for transient acquisition.
`--frames` saves headless snapshots instead of requiring a viewer.

Next: inspect finger seating and commanded-versus-measured pinch pose for the
two frozen failure placements. Test a targeted acquisition correction on new
placements, without selecting parameters using this held-out set. Only after a
scripted candidate passes should a serialized, explicit training preset be added
and a short fresh-policy pilot considered. Existing training commands still use
the original controller. No physical-robot approval follows from these results.

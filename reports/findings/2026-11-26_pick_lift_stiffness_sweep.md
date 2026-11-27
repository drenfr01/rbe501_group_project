# Pick-and-lift: measured-stiffness comparison (2026-11-26)

## Question and scope

Do the existing scripted and learned lift controllers continue to work when
the measured arm stiffness replaces the original gains? This is an evaluation
on the integration branch, not retraining or physical-robot operation.

**This is not the full calibrated reaching servo model.** Only measured arm kp
and the corresponding damping-ratio-preserving velocity-gain rescaling are
enabled. Reaching stiction, delay, encoder quantization, relative command clip,
and real joint-window wrappers are absent. Gripper dynamics and contact
friction remain the existing pick-and-lift assumptions. Pan and wrist-roll kp
are marked assumed in the calibration JSON, rather than directly measured.

## Protocol

- Code base: integration commit `3918166`, plus the evaluation script and tests
  added in this working tree. No controller, model, or preset implementation
  changes were made for this experiment.
- Four fixed masses: 5, 10, 15, 20 g; 25 mm cube.
- Same 20 placement seeds, 20000–20019, for each condition.
- Two dynamics settings and three controllers: 24 conditions, 480 episodes.
- Learned policies: existing seed-0 fixed-10-g and random-5–20-g SAC artifacts,
  deterministic inference using their matching frozen normalizers.
- Episode: 200 control steps, about 6.6 s. Success is the existing final-state
  criterion: both pads touching and cube lift at least 4 cm.
- Record final and maximum lift, transient/final success, any grasp, MuJoCo
  warnings, and robot/table contact at each control-step boundary. Contact
  sampling can miss contacts that exist only within physics substeps.
- Model, calibration, checkpoint, normalizer, and run-info hashes were identical
  before and after evaluation. Raw data records hashes, versions, and configs.

Reproduce in a **new** output directory (the script refuses overwrite):

```powershell
uv run --locked python scripts/sim2real/analyze/pick_lift_calibration.py --episodes 20 --out experiments/pick_lift_stiffness_repeat
```

## Final success counts (each cell is out of 20)

| Dynamics | Controller | 5 g | 10 g | 15 g | 20 g | Total |
|---|---|---|---|---|---|---|
| Original | Scripted | 20 | 20 | 20 | 20 | 80/80 |
| Original | Fixed-10-g SAC | 19 | 20 | 20 | 19 | 78/80 |
| Original | Random-5–20-g SAC | 19 | 19 | 19 | 19 | 76/80 |
| Measured stiffness only | Scripted | 20 | 20 | 20 | 18 | 78/80 |
| Measured stiffness only | Fixed-10-g SAC | 19 | 20 | 20 | 19 | 78/80 |
| Measured stiffness only | Random-5–20-g SAC | 17 | 18 | 18 | 19 | 72/80 |

The scripted controller loses two 20 g grasps under changed dynamics. The
fixed-mass policy retains its aggregate success count, although failure
placements change. The random-mass policy has four additional failures across
the sweep. Twenty placements and one learned seed per run do not establish
statistically significant differences or generalization to other workloads.
They do not demonstrate an advantage over the scripted baseline.

Measured-stiffness mean final lifts (cm, including failures):

| Controller | 5 g | 10 g | 15 g | 20 g |
|---|---|---|---|---|
| Scripted | 6.37 | 6.29 | 6.27 | 5.64 |
| Fixed-10-g SAC | 6.18 | 6.42 | 6.21 | 5.85 |
| Random-5–20-g SAC | 6.16 | 7.11 | 7.46 | 7.90 |

A larger lift does not itself mean better or safer control. For example, the
measured-stiffness random-mass policy at 5 g, placement 20008, reaches success
transiently (maximum lift 7.47 cm) but fails at episode end. This is counted as
a failure. No other episode loses a previously observed success in this sweep.

## Contact and numerical findings

All 480 episodes remained finite and recorded zero MuJoCo warnings. This does
not imply collision-free operation.

Robot/table contact occurred in 176/240 original-dynamics episodes and 238/240
measured-stiffness episodes. The task's success flag does not reject table
contacts. No collision settings were weakened or disabled for this test.

A separate four-episode audit at 10 g, placement 20000, inspected the scripted
and fixed-10-g policies under both dynamics:

- Original scripted: no robot/table contact in that placement.
- Measured-stiffness scripted: both fingertip pads contact the table, maximum
  solver penetration about 0.73 mm.
- Original fixed-mass SAC: both fingertip pads contact the table, maximum
  penetration about 0.72 mm.
- Measured-stiffness fixed-mass SAC: both pads contact the table, maximum
  penetration about 1.39 mm. These contacts occur during descent/closing.

Those audit findings identify the geoms for these four episodes only, not
every table contact in the full sweep. Penetration is a contact-solver quantity,
not a measurement of force or real material deformation.

## Artifacts and validation

- `experiments/2026-11-26_pick_lift_stiffness_check/results.json`: all summaries,
  480 episode records, configs, versions, and input hashes.
- `experiments/2026-11-26_pick_lift_stiffness_check/contact_audit.json`: the
  four audited episodes, pad names, contact step ranges, and penetration.
- `scripts/sim2real/analyze/pick_lift_calibration.py`: evaluation-only runner.
- `tests/test_pick_lift_calibration_sweep.py`: unchanged-condition settings,
  final-versus-transient success accounting, and missing-data rejection.
- Full suite after adding the runner: **359 passed, 1 skipped** (optional
  LeRobot hardware configuration check). No new training job was started.

## Next experiment

Do not treat these results as approval for physical deployment or jump straight
to retraining on a supposedly fully calibrated model. First reproduce the
failed placements and pad/table contacts in the viewer. Then define a
manipulation-specific servo-effects integration that preserves cube/gripper
state, check its timing and units, and compare it separately with this
stiffness-only baseline. Any contact-clearance or reward changes should be
separate experiments, retaining the original checkpoints and results.

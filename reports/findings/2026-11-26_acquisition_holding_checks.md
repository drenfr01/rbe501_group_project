# Acquisition diagnosis and frozen-policy holding probes — 2026-11-26

Completed: freeze the working reference, investigate its shared acquisition
failure, and compare scripted versus limited-RL holding under controlled forces.
**No training or controller/model/gain changes.**

Two findings matter: the bad placement has premature stationary-finger contact
during approach; the limited RL policy shows a very small numerical cube-motion
benefit under the tested forces, with no retention advantage and more TCP motion.
These findings do not justify replacing the script or launching full training.

## 1. Frozen baseline

Base commit: `4b780ae`, integration/calibrated-pick-lift. The evaluation records
before/after SHA256 hashes for the source MJCF, calibration, original saved
policies, v1 pilot, v2 policy/normalizer/configuration and relevant controller
source. All match. The script, v2 envelope, task, rewards, gains, cube geometry,
contact settings and sensing/servo effects remain unchanged. Main is untouched.

Artifacts: `experiments/2026-11-26_acquisition_holding_checks/`. The protocol was
saved before acquisition/holding evaluation. No force, threshold or model was
tuned using the new results. Binary policy files remain local under checkpoints.

## 2. Shared acquisition failure: early contact, then poor seating

Sixteen scripted episodes: seeds 68000 (working control) and 68006 (shared failure),
5/10/15/20 g, stiffness alone versus stiffness plus nominal arm effects. Trace
the first 100 steps, signed pad/cube distances, pad contact forces and locations,
physical pinch/cube positions and jaw angle. Retain the final task outcome.

| Placement | Dynamics | Final successful lifts |
|---|---|---:|
| 68000 | Stiffness only | 4/4 masses |
| 68000 | Arm effects | 4/4 masses |
| 68006 | Stiffness only | 4/4 masses |
| 68006 | Arm effects | 2/4 masses (10 g, 20 g) |

No robot/table contact episodes or MuJoCo warnings in these probes.

At seed 68006 with arm effects:

- The **stationary finger first contacts the cube at step 9**, during approach.
  Contact locations are near the cube's upper side edge. At 5 g, cube XY shifts
  from approximately (56.85, 19.05) mm to (49.73, 21.29) mm by step 40: about
  7.5 mm horizontal displacement before closing.
- At step 70, the physical pinch is about (7.3, -3.1, 6.6) mm relative to the
  cube: laterally offset and too high. The moving pad first contacts at 80.
- At 5 g, both pads touch at steps 84–87, then the fixed pad loses contact at
  88. The cube moves sideways during closing; its Y coordinate grows from
  about 21 mm before close to 80 mm by step 100. The jaw reaches its -10 degree
  close command without retaining the object. The lift gate never releases.
- At 15 g, two-pad contact lasts steps 84–88 before the cube is pushed away.
- At 10/20 g, contact persists and the grasp succeeds despite imperfect seating.

In contrast, stiffness-only seed 68006 has no early approach contact: moving-pad
contact starts at 81 and fixed-pad contact at 82. Both remain through closing;
the pre-close pinch is approximately (0.1, 0, -3.3) mm relative to the cube.
The working seed 68000 with arm effects first touches the fixed pad during
descent at step 47 and shifts the cube only about 0.47–1.07 mm before close.

This isolates an approach-path/tracking/contact interaction, not an unreachable
ideal target or simple insufficient lifting capacity. Early contact alone is not
a complete causal explanation: heavier and intermediate masses have different
closing/contact outcomes, and all four masses experience the approach disturbance.
The measured pose error and subsequent asymmetric seating also matter. Contact
forces are solver outputs, not measured physical gripper forces.

**Next acquisition experiment:** an explicit raise-before-traverse approach and
an actual-pose alignment check before closing. Test whether the cube stays still
during approach, both pads seat correctly, and table clearance is maintained.
Keep this separate and opt-in; do not restore unrestricted descent residuals.

## 3. Frozen holding disturbance protocol

One saved v2 training seed; deterministic inference, no weight updates. Ten fresh
placement seeds **69000–69009**, four fixed masses (5/10/15/20 g), both controllers.
Nine conditions: no force plus +/-world X/Y pulses at 0.1 N and 0.2 N. Total:
**720 evaluation episodes**.

Forces act on the cube COM in world coordinates, with no added torque, for three
33 ms control periods: **99 ms**, from t=4.950 to 5.049 s. Apply a pulse only if
the preceding ten steps all satisfy the unchanged two-pad/4 cm success criterion.
Restore the original applied wrench after every period, including exceptions.
Ineligible acquisitions remain in overall outcomes; they are not counted as
successful disturbance trials.

Use true simulated cube/pinch positions for evaluation, separate from the delayed
and quantized observations provided to the controller. Measure final-window RMS
displacement from the controller's own pose just before the pulse, over the last
30 steps (0.99 s). This measures held-pose motion, **not** IK target error or
absolute world positioning accuracy. Include dropped objects in metrics rather
than censoring them. Record peak cube displacement and relative cube/pinch slip too.

Primary comparison, frozen beforehand: subtract each controller's matched no-force
cube RMS, then compute RL-minus-script force-associated RMS. Bootstrap placement
cluster means with 10,000 resamples, RNG seed 0. A negative difference favors RL.
The numerical benefit rule requires the 95% interval below zero, no worse eligible
retention and no additional table-contact episodes.

## Holding outcomes

| Outcome | Script | Limited RL |
|---|---:|---:|
| Stable pre-pulse holds, each condition | 39/40 | 39/40 |
| Retained successful holds after nonzero forces | **312/312 eligible** | **312/312 eligible** |
| Final successes across all nine conditions | 351/360 | 351/360 |
| Table-contact episodes | 0/360 | 0/360 |
| MuJoCo warning episodes | 0/360 | 0/360 |

The ineligible case is 5 g / placement 69005, repeated across force conditions.
It fails acquisition before a pulse is applied; this run does not establish that
its detailed failure mechanism is identical to seed 68006. There are 312 common
eligible disturbed pairs, not 312 independent placements. Retention has a ceiling
in this test: neither controller drops an already-stable hold at these force levels.

Across eligible nonzero-force cases (descriptive means):

| Metric | Script | Limited RL |
|---|---:|---:|
| Final-window cube displacement RMS | 0.3628 mm | 0.3379 mm |
| Peak cube displacement | 0.6703 mm | 0.6590 mm |
| Final-window physical pinch displacement RMS | 0.00095 mm | 0.05979 mm |
| No-force cube RMS (separate control condition) | 0.3630 mm | 0.3526 mm |

The primary placement-cluster mean adjusted difference is **-0.01430 mm**, with
95% bootstrap interval **[-0.01901, -0.00965] mm**. It meets the predeclared
numerical rule: about **14 micrometers** less force-associated cube tail motion.
Raw forced cube RMS differs by about 25 micrometers; part of that difference
already exists in the no-force condition.

### Interpretation and limits

This is a small, measurable **simulation** result, not established practical or
hardware improvement. The policy's pinch point moves more than the script's,
despite slightly lower cube RMS. Cube stability and TCP precision are different
objectives; a universal holding-precision improvement cannot be claimed.

The interval resamples only ten placement clusters and one frozen trained seed.
It does not include training-seed, model/calibration or hardware uncertainty.
The contact/gripper model remains uncalibrated. These moderate, short forces do
not locate the grip-failure boundary, and this experiment does not separate
feedback adaptation from effects of the policy's different holding posture.
No practical minimum benefit threshold was preregistered; do not introduce one
after seeing the result or describe 14 micrometers as validated robot accuracy.

## Validation and reproduction

**393 tests passed, one optional LeRobot test skipped.** New tests cover exact
pulse timing, force restoration after exceptions, expected impulse on an isolated
free cube, fixed-reference RMS, paired no-force subtraction and eligible pairing.
A complete zero-force rollout matches the existing baseline's final physical
state exactly. Source model, policy/configuration and controller hashes match.

Outputs include `acquisition.json`, all 720 holding episodes, grouped summary,
312 eligible paired results, assessment, interpretation, frozen protocol, source
provenance and input verification. Reproduce into a new output directory:

```powershell
uv run --locked python -m scripts.sim2real.analyze.acquisition_holding_checks --model-dir checkpoints/calibrated_lift_only_v2_pilot --out experiments/acquisition_holding_repeat
```

The v2 and earlier v1 companion files must exist locally for policy loading and
protected-input hashing; they are not uploaded by the code/results commit.

The next priority is to eliminate premature approach contact and verify actual
pre-close alignment on fresh placements. Preserve the script as the working
reference. Do not launch full training based solely on this small holding effect.

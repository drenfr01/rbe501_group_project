# Clearance-checked Cartesian descent — 2026-11-26

The selected Cartesian prototype **solves the measured pre-close cube-pushing
problem on the fresh batch**, but is not a better overall grasp controller yet.
With arm effects it lifts **79/80**, versus **80/80** for both existing references.
All Cartesian pre-close cube movement is below 0.16 mm and there are no observed
table or initial-approach contacts. Known 5 g regressions remain unresolved.

The independent batch meets the numeric held-out thresholds, but the prerequisite
known-regression/tuning gate fails. Therefore `overall_validated=false`; defaults,
saved policies and training presets remain unchanged. No training was started.

## New opt-in controller

Base commit `dc98d92`, integration/calibrated-pick-lift. `CartesianDescent` reuses
the existing raised approach, five-arm-joint position/approach-axis IK, joint
limits and gravity feedforward. Gripper control remains separate. Physical gains,
MJCF, collision fixes, cube, scene, reward and saved weights are unchanged.

Acquisition sequence:

1. Raise before traversing, as in the preceding experiment.
2. Align sensed pinch XY within 3 mm for three measurements while still high.
3. Advance a vertical Cartesian reference at fixed target XY in 2 mm increments.
   Hold the reference if measured vertical lag exceeds 8 mm. Use bounded tracking
   compensation, gain 0.2, +/-12 mm per axis, updated every two control periods.
4. Do not integrate vertical feedback while the reference is moving normally;
   compensate vertical error only when stalled or at the final target. Stop
   sub-tolerance feedback to avoid accumulating a downward bias during motion.
5. Sample proposed joint segments at fractions .25/.5/.75/1 for >=4 mm nominal
   robot/table-top clearance, checking open, midpoint and closed jaw postures.
   An optional variant also requires >=0.3 mm nominal robot/cube clearance while
   open. Use private MuJoCo scratch data, leaving physical state untouched.
6. Require sensed closing alignment (3 mm XY, 4 mm Z, 8 degree tilt) and sampled
   closing-jaw table clearance before closing. Require at least 25 closing steps
   as well as stable two-pad contact before lift; do not release lift immediately
   just because late closure happens to establish contact for five steps.

The new prototype uses cube-center pinch height, **0 mm**, rather than the old
-4 mm waypoint. This preserves more closing-jaw table clearance. The nominal
phase clock/horizon remains 40/30/30/100, 200 steps / 6.6 s; descent can continue
open into the close window, and actual lift can be delayed by closure settling.
If alignment does not complete by nominal lift time, abort open.

Trajectory, feedback, closing checks, settling and final height change together:
this is **not a pure path-only causal ablation**. No arbitrary full 6-DOF pose
constraint or actuator gain tuning was added.

These are sampled nominal geometry checks, not swept-volume certification or
hardware safety guarantees. They do not predict servo/contact dynamics. The
checker copies true simulated object pose; availability/accuracy of physical
object sensing is not validated. Existing delayed commands can still move after
an abort. Recorded `clearance_rejections` include nonconverged IK proposals too.

## Development before freezing

An early known-case prototype still touched the table: it accumulated vertical
feedback during motion and checked only the open-jaw geometry. That version was
rejected. Before independent evaluation, add anti-windup, sampled closing geometry,
the higher waypoint and a closing dwell. Known-case tests still show 5 g failures
despite no table contacts. Do not interpret contact avoidance as grasp success.

## Frozen comparisons

Five variants: original baseline, raise-only reference, Cartesian without feedback,
Cartesian table-only checks, and Cartesian table-plus-cube checks. Known seeds
68000/68006/69005, four masses, two dynamics modes: 24 episodes per variant.
Tuning seeds 74000–74004: 40 episodes per variant. Total **320** regression/tuning
episodes. Freeze selection before fresh seeds **75000–75019**: three controllers,
four masses and two modes, **480 held-out episodes**.

Gate unchanged from the preceding acquisition test: >=95% final lift success at
every mass/mode, no observed table/approach contacts or warnings, <=5% of episodes
with >1 mm pre-close XY cube movement, and known regressions must lift. Aborts
count as failures. Placements are reused across masses/modes, not independent
random samples for every episode.

| Variant | Known successes /24 | Tuning successes /40 | Tuning table contacts |
|---|---:|---:|---:|
| Baseline | 21 | 40 | 0 |
| Raise only | 22 | 40 | 0 |
| Cartesian, no feedback + cube checks | 8 | 0 | 0 |
| Cartesian, table checks | 22 | 40 | 0 |
| Cartesian, table + cube checks | 18 | 4 | 0 |

The strict cube-check variants frequently cannot finish within the window;
36/40 tuning episodes abort in the full-check variant and 40/40 without feedback.
This reflects this prototype's checking, IK and deadlines, not proof that the
physical grasp is impossible. The table-only Cartesian variant is selected for
exploratory validation, with `passed_tuning_gate=false` because it does not repair
the known 5 g cases. The no-feedback row also has cube checks, so it does not
isolate feedback's contribution in the selected table-only configuration.

## Held-out results

Each cell is final successful lifts out of 20:

| Mass | Baseline, stiffness | Raise, stiffness | Cartesian, stiffness | Baseline, effects | Raise, effects | Cartesian, effects |
|---|---:|---:|---:|---:|---:|---:|
| 5 g | 20 | 20 | 20 | 20 | 20 | 19 |
| 10 g | 20 | 20 | 20 | 20 | 20 | 20 |
| 15 g | 20 | 20 | 20 | 20 | 20 | 20 |
| 20 g | 20 | 20 | 20 | 20 | 20 | 20 |
| Total | **80** | **80** | **80** | **80** | **80** | **79** |

Arm-effect acquisition metrics, out of 80 cases per controller:

| Metric | Baseline | Raise only | Cartesian |
|---|---:|---:|---:|
| Pre-close cube movement >1 mm | 34 | 34 | **0** |
| Median pre-close XY movement | 0.922 mm | 0.912 mm | **0.00042 mm** |
| Maximum pre-close XY movement | 2.300 mm | 2.305 mm | **0.156 mm** |
| Table-contact episodes | 0 | 0 | 0 |
| Initial-approach contact episodes | 0 | 0 | 0 |

With stiffness alone, Cartesian maximum pre-close movement is 0.058 mm and all
80 lifts succeed. No formal-evaluation MuJoCo warnings. This fresh batch happens
not to expose the old early-approach contact; earlier reports contain that evidence.

### Remaining failure is retention, not descent movement

New 5 g/75003 is a regression: pad contact begins at step 89, lift is released,
and contact is lost at 106 during early lift before achieving the 4 cm threshold.
The guard aborts and the cube returns to the table. It was not an alignment abort.
The known 5 g cases 68006/69005 also remain failures; known-case probes show contact
can disappear while the arm is still waiting for closure settling, before lift.

Thus improved descent is not sufficient to fix light-cube gripping/retention.
Changes in final height and contact seating may matter; this experiment does not
identify one unique cause. Do not retune this controller on held-out seed 75003.

## Validation, artifacts and reproduction

**403 tests passed, one optional LeRobot test skipped.** Initial tests failed before
implementation; coverage includes gradual waypoint progression, bounded feedback,
rejecting zero geometry samples, holding the prior target after rejected proposals,
sampled closing geometry, and requiring closing dwell despite stable contacts.

Physics-step audit: baseline/raise/Cartesian, 5/15 g, seeds 68006/75003, twelve
episodes / **79,200 physics steps**, zero table-contact substeps. Cartesian
pre-close XY motion in the audited subset is at most 0.140 mm. This audit is a
subset, not a continuous-safety guarantee for all formal episodes.

`experiments/2026-11-26_cartesian_descent/` preserves protocol, all comparisons,
frozen selection, explicit failed overall assessment, source provenance, input
verification, audit, successful 10 g replay and failed 5 g replay. Original files
and v2 policy companions match their frozen hashes. No saved policy was executed
on this changed controller; existing RL results do not automatically transfer.

Repeat with a new output directory:

```powershell
uv run --locked python -m scripts.sim2real.analyze.cartesian_descent_check --out experiments/cartesian_descent_repeat
```

Replay the frozen experimental candidate (always saves phase PNGs):

```powershell
uv run --locked python -m scripts.sim2real.analyze.cartesian_descent_check --replay experiments/2026-11-26_cartesian_descent/frozen_selection.json --mass-kg 0.005 --placement-seed 75003 --viewer --out experiments/cartesian_descent_viewer
```

Next isolate closing and retention with a fixed aligned arm, before adding lift:
record both pad forces/contact locations, jaw motion, cube rotation and sliding.
Then test a controlled closing profile and early-lift retention on new placements.
Keep physical gains and the old baseline fixed; do not start full RL training or
promote this prototype while its known regressions remain unresolved.

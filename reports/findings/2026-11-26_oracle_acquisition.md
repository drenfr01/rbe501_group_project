# Orientation-aware acquisition: one prototype and one independent validation

**Cube orientation substantially improves rotated grasps, but this prototype
does not pass the no-regression gate.** Independent lifts improve from 167/280
for the original script to 268/280 for the oracle prototype. However, nominal
controls fall from 40/40 to 36/40, and the recorded workspace grid falls from
180/180 to 152/180. No prototype table/initial-approach contacts or MuJoCo warnings
were sampled. Keep the working baseline and saved policies unchanged; do not
promote the controller or start training from this result.

The agreed scope is complete: one developed/frozen controller, a recorded-case
batch and one independent validation. The candidate was not retuned after either
batch. The remaining issue is a specific acquisition timing/settling regression,
not a reason to begin an open-ended benchmark search.

## What was implemented

Two new opt-in runtime modules, without edits to existing runtime files:

- `kinematics/oracle_jaw_ik.py`: weighted damped least-squares IK for pinch-point
  position and TCP X/jaw-axis alignment with a cube face. The cube is symmetric
  under 90-degree yaw rotations; choose the equivalent face axis nearest the
  nominal reachable jaw direction, rather than unnecessarily reversing the jaw.
- `control/oracle_acquisition.py`: a separate Cartesian acquisition controller
  and factory. Read the true MuJoCo cube yaw once per acquisition. Add a measured
  jaw-alignment check to the original closure readiness conditions.

The solver uses `mj_jacSite` through the existing pinch solver: axis derivative
is `-[axis] J_omega`, while the pinch translation Jacobian includes the site-to-
pinch offset. Position has weight 1; jaw direction has weight 0.03 m/rad; a
weak approach preference has weight 0.001 m/rad. This constrains the direction
needed for side grasping instead of demanding an arbitrary full six-dimensional
pose from five arm joints. The gripper command is kept out of arm IK.

Frozen parameters are damping 0.005, at most 200 iterations per oriented start,
position tolerance 1 mm, IK jaw tolerance 1 degree and measured closure jaw
tolerance 5 degrees. The solver accepts approach tilt up to 25 degrees, but the
existing **stricter 8-degree measured closure tilt gate remains in force**.
All original measured XY/Z readiness thresholds, three stable samples, closing
dwell, contact guard, 4 mm nominal table-clearance checks, gravity compensation,
0-degree closing endpoint, cube collisions, gains and 200-step horizon remain.

During bring-up, using the same 5-degree threshold for ideal IK and measured
closure caused a 45-degree case to stop at about 6 degrees measured jaw error.
Tightening the ideal solve to 1 degree leaves tracking margin without relaxing
the closure gate. Recorded seed 82000 / 10 g / offsets 0, 30, 45 degrees was
used for bring-up; those cases are development evidence, not held-out validation.
The final settings were frozen before the full batches and not changed afterward.

This is a **privileged orientation oracle**, not a perception system, new policy
observation or retrained neural controller. All evaluation actions are zero
residual, comparing two scripted acquisition pipelines. No saved policy was run
through the changed controller as though it had been trained with it.

## Frozen comparisons

Integration base commit: `e1f516d`. Original baseline results and input hashes
from the coverage audit were checked before reusing its recorded-case reference.

| Test | Original script | Oracle prototype |
|---|---:|---:|
| Recorded nominal workspace grid | 180/180 | 152/180 |
| Recorded yaw cases, seeds 82000–82009 | 128/280 | 280/280 |
| Independent yaw cases, seeds 85000–85009 | 167/280 | 268/280 |
| Independent nominal-yaw controls (subset) | 40/40 | 36/40 |

The grid is five radii 0.14–0.22 m, nine azimuths -40 to +40 degrees, four masses
5/10/15/20 g, reset seed 83000. Yaw cases use seven offsets -45/-30/-15/0/+15/
+30/+45 degrees at ten normal sampled centers and four masses. Independent
validation uses a new placement batch, the same fixed nominal arm effects and
deterministic scripted action. Counts repeat centers across masses/angles;
they are not independent placements. The +/-45 degree conditions are equivalent
orientations of a square cube.

The 460 prototype development episodes and 560 independent paired episodes
are **1,020 new matrix episodes**. The historical reference is not counted as
new evaluation. Additional bring-up, image replays and timing diagnosis are
separate from these denominators.

| Independent yaw offset | Original | Oracle |
|---:|---:|---:|
| -45 degrees | 12/40 | 40/40 |
| -30 degrees | 12/40 | 32/40 |
| -15 degrees | 40/40 | 40/40 |
| 0 degrees | 40/40 | 36/40 |
| +15 degrees | 39/40 | 40/40 |
| +30 degrees | 12/40 | 40/40 |
| +45 degrees | 12/40 | 40/40 |

Among nonzero yaw cases, success rises from 127/240 to 232/240: +43.75 percentage
points, with placement-cluster bootstrap 95% interval [+25.0, +60.42] points
(10,000 resamples, seed 0). Ten placement clusters only; this does not capture
hardware/model uncertainty. Improved rotated success does not cancel the nominal
regressions. The -30-degree group also misses the frozen >=95%-per-yaw gate.

All 740 prototype episodes have zero sampled robot/table contacts, initial
approach contacts and MuJoCo warnings. The independently reevaluated original
script has seven table-contact episodes, all at 10 g, in rotated conditions.
Counts are at control boundaries, not a physics-substep safety certification.

All 12 independent prototype failures abort before closing with
`alignment_timeout`: seed 85000 at -30 degrees and seed 85007 at -30/0 degrees,
each repeated at four masses. Four of those are new paired failures where the
original script succeeds (seed 85007, nominal yaw). None of these final failures
occurs after grasp establishment.

## Achieved alignment and the nominal regression

For the 268 independent prototype episodes that reach closure, last acquisition
IK residuals average **0.725 mm position / 0.689 degrees jaw direction**, maxima
**0.999 mm / 0.978 degrees**. These refer to the last corrected Cartesian target,
not necessarily the exact uncorrected cube center.

Measured closure position error relative to the cube center averages **2.041 mm**
(maximum 2.980 mm), measured jaw error averages **2.083 degrees** (maximum
3.066 degrees), and measured approach tilt averages **2.381 degrees** (maximum
2.725 degrees). These are conditional on reaching closure; failed cases are
retained in episode/success counts and their separate diagnostics.

The 28 workspace regressions repeat seven azimuths at radius 0.20 m across four
masses. A paired trace at radius 0.20 m, azimuth -40 degrees, 10 g reproduces:

- Original reference reaches `at_goal` at step 75, begins the stable-alignment
  count at 77 and starts closing at 79.
- Oracle reaches `at_goal` at 97. Its measured Z error is still 7.08/5.92/4.70 mm
  on steps 97/98/99. Position becomes acceptable at step 100, starting the
  stable count at one.
- The unchanged clock enters lift at 101. Because three stable samples have
  not authorized closing, the acquisition guard latches `alignment_timeout`.

Thus a terminal 3-D position norm around 4.7 mm does not itself prove a permanently
unreachable grasp: the actual gate uses separate XY and Z bounds, and this case
becomes ready too late. The new solver changes the motion/settling schedule.
This mechanism is directly traced for that example; other aborts remain recorded
as alignment timeouts and should not all be assumed identical without traces.

## Decision and next targeted change

`assessment.json` rejects the independent validation; `summary.json` also records
the failed development workspace-preservation condition. The frozen requirements
were no new paired failures, all nominal controls retained, >=95% per yaw,
positive rotated-success gain and zero candidate contacts/warnings. Only the
orientation improvement and sampled contact checks pass.

The next focused change should address **readiness versus the fixed phase clock**:
allow a bounded acquisition-settling transition while preserving the existing
numeric alignment/contact gates and enough time for closing and lifting. Test
the recorded timing regression first, then repeat the frozen no-regression
comparison. Do not loosen tolerances, increase actuator gains or remove collision
checks to conceal the timeout. That change was not implemented in this experiment.

Orientation-aware acquisition is feasible and useful in simulation, but is not
ready to become a training baseline. After it preserves nominal behavior, decide
how hardware supplies orientation and how a new policy is allowed to use it.
No observation/reward changes or RL training are justified by this candidate yet.

## Validation and artifacts

**429 tests pass, one optional LeRobot test skips.** New tests were introduced
before the IK implementation and cover square symmetry, angular Jacobian versus
finite differences, a recorded rotated-pose solve, fixed gripper command and
oracle reset. Compiled gain/bias/force/joint/collision/mass/friction/gravity arrays
and existing acquisition safeguards match the baseline. Assessment tests reject
nominal regressions despite large rotated gains and reject incomplete matrices.

New files are the two runtime modules above, evaluation runner
`scripts/sim2real/analyze/oracle_acquisition_check.py`, two test files, this report,
and evidence under `experiments/2026-11-26_oracle_acquisition/`. Documentation
links the opt-in result. Existing runtime files, main branch, checkpoints,
training presets and user-local CUDA dependency edits remain unchanged.

Evidence includes the preregistered protocol, development/historical reference,
full independent outcomes, assessment, summary, source/software provenance,
before/after protected hashes, comparison figure and plotting source, repaired
45-degree replay, nominal regression replay and paired timing trace. Binary
policy artifacts remain local and are not included in this commit.

```powershell
uv run --locked python -m scripts.sim2real.analyze.oracle_acquisition_check --model-dir checkpoints/calibrated_cartesian_zero_v3_pilot --out experiments/oracle_acquisition_repeat
```

Use a new output directory. This comparison requires locally retained previous
checkpoint companions and the original coverage evidence for protected-input
checks. It never calls a training routine.

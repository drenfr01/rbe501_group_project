# Bounded acquisition settling repairs the orientation prototype

**The timing repair passes the frozen simulation gate:** 460/460 recorded
regression cases and 280/280 fresh yaw cases succeed, including all nominal
controls. The original radial script completes 140/280 on the same fresh set.
The new candidate has zero sampled robot/table contacts, early approach contacts
or MuJoCo warnings. Selected repaired/fresh cases also pass an every-physics-step
contact audit.

This completes the targeted timing change and independent retest. No additional
controller tuning or training was performed. The previously working controllers,
orientation prototype, model, gains, policies, observations and training presets
remain unchanged. Base integration commit: `8af5141`.

## What changed

New opt-in module `control/timed_oracle_acquisition.py` reuses the frozen oracle
orientation solver and Cartesian approach. It changes two things together:

1. At the nominal lift boundary, allow a bounded continuation of acquisition
   until measured readiness authorizes closing. Continue the close phase until
   the existing real-time closing dwell has elapsed.
2. Synchronize the lift guard with that actual phase. Merely extending the
   controller clock would leave the old step-100 guard freezing the arm and
   commanding closure while acquisition is still settling.

Real controller time keeps increasing; no step counter is paused or rewound.
The original episode remains **200 control steps, 6.6 seconds**. Frozen timing
configuration uses zero-based transition indices:

- Alignment deadline 120: at most 20 extra control periods (660 ms) beyond the
  nominal step-100 lift transition. If closing is still unauthorized, the existing
  alignment timeout aborts with the jaw open.
- Preserve the original 25-period closing dwell (at least 825 ms); calculate it
  from actual closing start, not a virtual phase counter.
- Lift-release deadline 150: absent stable contact, abort open rather than wait
  indefinitely. Reserve at least 50 control periods (1.65 seconds) for lift
  inside the original horizon.

All measured readiness gates remain: XY <=3 mm, absolute Z <=4 mm, approach
tilt <=8 degrees, jaw-axis error <=5 degrees, three consecutive aligned samples,
five consecutive two-pad contact samples for release, closing dwell and the
existing contact-loss latch. Position IK tolerance remains 1 mm and ideal jaw
solve tolerance 1 degree. Existing gains, force/position limits, gravity
compensation, cube collisions, 0-degree closing endpoint and 4 mm nominal
clearance checks remain intact. Additional time does not bypass failed readiness.

The cube-orientation input remains privileged initial MuJoCo yaw. This is a
scripted oracle/controller validation, not a new neural policy or hardware
perception implementation.

## Recorded regression and independent validation

The protocol was saved before either matrix. The recorded grid timeout was used
for bring-up; no new placement results were used for tuning.

| Comparison | Previous orientation prototype | New bounded timing |
|---|---:|---:|
| Same nominal workspace grid | 152/180 | **180/180** |
| Same recorded yaw cases, seeds 85000–85009 | 268/280 | **280/280** |

The grid is five radii 0.14–0.22 m, nine azimuths -40 to +40 degrees, four
masses 5/10/15/20 g, reset seed 83000. Recorded yaw cases use seven offsets
-45/-30/-15/0/+15/+30/+45 degrees at ten sampled centers and four masses.
All 460 recorded cases now lift; latest recorded closing start is step 106.

Independent seeds **86000–86009**, not used in bring-up or recorded testing,
repeat all seven yaw offsets and four masses in the same nominal arm-effects
mode. Both pipelines are freshly evaluated using zero residual actions.

| Independent test | Original radial script | Orientation + bounded timing |
|---|---:|---:|
| All yaw cases | 140/280 | **280/280** |
| Nominal-yaw controls | 40/40 | **40/40** |
| Nonzero yaw cases | 100/240 | **240/240** |
| Robot/table contact episodes | 8/280 | **0/280** |
| Initial approach contact episodes | 0/280 | **0/280** |
| MuJoCo warning episodes | 0/280 | **0/280** |

Every candidate yaw/mass cell passes 10/10. There are no new paired failures;
all nominal behavior sampled in the recorded grid and fresh controls is retained.
Nonzero-yaw success gain is **+58.33 percentage points**, placement-cluster
bootstrap 95% interval **[+44.56, +67.50] points** (10,000 resamples, seed 0).
Repeated masses and angles are clustered by ten placement centers, not treated
as independent placements. +/-45 degrees are equivalent square-cube orientations.
Finite 100% success does not establish a zero underlying failure probability.

The fresh original-script comparison measures the benefit of **orientation plus
timing together**. The recorded comparison isolates the timing change against
the prior orientation prototype. Do not attribute the entire fresh gain to timing
alone or to RL learning. No trained policy was evaluated through a changed
controller as though it were a matching saved-policy environment.

There are 1,020 matrix episodes: 460 recorded candidate episodes plus 280 fresh
episodes for each pipeline. No horizon extension, gain adjustment, tolerance
relaxation or controller retuning followed the frozen protocol.

## Readiness and actual phase evidence

The recorded 10 g / radius 0.20 m / azimuth -40 degrees / seed 83000 case
previously became aligned on step 100, leaving only one of three required stable
samples before the old lift clock aborted. It now starts closing at **102** and
releases lift at **128**, succeeding by step 200 with 72 later control periods.

The recorded -30-degree / 10 g / seed 85007 case starts closing at **104** and
releases lift at **130**, then succeeds. Replays explicitly log actual phase,
real step counter, alignment count, close start, retained contact and abort
reason; old nominal phase labels alone would be misleading.

In all 280 fresh candidate episodes, acquisition IK position error averages
**0.782 mm** (maximum 0.999 mm), and jaw-axis error averages **0.705 degrees**
(maximum 0.998 degrees). These refer to the last corrected Cartesian target.
Measured position error relative to the cube at closure averages **2.080 mm**
(maximum 3.069 mm), and measured jaw error averages **2.216 degrees** (maximum
3.034 degrees). The measured position gate uses separate XY and Z limits, not
a 3 mm threshold on the full 3-D norm. All episodes reach closure, so no failed
candidate cases are excluded from these fresh diagnostic means.

## Physics-step contact audit

An additional read-only `mj_step` wrapper audits four selected pose conditions
at all four masses: the repaired grid case, recorded seed 85007 at -30 degrees,
and fresh seed 86000 at +/-45 degrees. These are **16 selected episodes**, not
16 independent new placements. The wrapper restores MuJoCo's original function
even on exceptions and changes no commands or physics state.

All 16 lift, with **zero robot/table contact substeps and zero initial-approach
cube-contact substeps across 105,600 physics steps**. Maximum pre-close XY cube
movement in this audit is 0.0463 mm. No warnings. This supplements the control-
boundary counters on the full matrix; it does not certify all matrix cases at
substep resolution or constitute hardware safety validation.

## Decision

`assessment.json` passes the combined gate: complete independent pairing,
all recorded regressions pass, all nominal controls pass, >=95% per yaw,
positive rotated-success interval, no new paired failures and zero candidate
sampled table/approach contacts or warnings. The frozen setup is a validated
**experimental simulation controller**, not an automatic replacement for defaults.

The identified acquisition/timing blocker is resolved on the tested sets. Stop
adding unrelated diagnosis before the next decision. Preserve this exact setup
as a versioned baseline if it will support a new training run. The success gain
comes from the script and privileged yaw; it is not evidence that the existing
lift-only RL residual adds value or that a longer training job is needed.
Hardware orientation input and the intended learning objective still need an
explicit choice before expanding policy observations or acquisition authority.

## Files, tests and reproducibility

New runtime file: `src/rbe_501_so101/control/timed_oracle_acquisition.py`.
New runner: `scripts/sim2real/analyze/timed_oracle_check.py`.
New tests: `tests/test_timed_oracle.py`. Existing runtime files are unchanged.

Tests were written first and failed before implementation. **433 tests pass,
one optional LeRobot test skips.** New tests cover deadline boundaries, real-time
closing dwell, the actual repaired grid regression inside 200 steps, failed
alignment aborting open without clock freezing, and missing contact never
releasing lift. Full protected source/model/calibration/policy/statistics hashes
match before and after the comparison. `git diff --check` passes.

Evidence under `experiments/2026-11-26_timed_oracle/`: frozen protocol, all episode
outcomes, summary, acceptance assessment, source/software provenance, input and
artifact hashes, comparison figure and source, two phase-aware replays and the
physics-step audit with its source. The selected saved checkpoint companions
remain local and ignored by Git. User-local CUDA dependency edits and personal
replay directories are untouched. No training or dependency synchronization ran.

```powershell
uv run --locked python -m scripts.sim2real.analyze.timed_oracle_check --model-dir checkpoints/calibrated_cartesian_zero_v3_pilot --out experiments/timed_oracle_repeat
```

Use a new output directory. Matching baseline configuration and locally retained
prior artifacts are required for source/checkpoint verification; the runner
never loads a neural policy or calls training.

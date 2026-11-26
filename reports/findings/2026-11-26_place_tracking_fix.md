# Contact-aware tracking repairs the last placement timeout

**The complete scripted task now passes the frozen checks:** 240/240 recorded
conditions and 120/120 new conditions complete reach, grasp, lift, carry, place,
release and retreat. There are no new paired failures, sampled robot/table
contacts or MuJoCo warnings. The formerly failing 5 g / +30-degree / seed 89000
case now releases onto the table and leaves the cube at rest.

This fixes the identified final-lowering blocker on the tested sets. No RL
training, gain/model change, threshold relaxation or phase-budget increase was
used. Existing sequence variants, acquisition, saved policies and old results
remain available unchanged. Integration base commit: `ec7ed81`.

## Diagnosis

The old supported-placement candidate reproduced `lower_timeout` at seed 89000,
5 g and +30-degree cube yaw. It retained its grasp but stopped about 3.36 mm above
resting height and 5.25 mm off horizontally. Final numerical IK residual was
about **0.281 mm** with **0.552 degrees** jaw error. The Cartesian reference had
reached its requested goal, yet measured pinch tracking still lagged by roughly
(-3.95, +3.47, -6.37) mm at the last step. The small IK residual therefore did
not imply successful physical touchdown.

The original release guard correctly refused to open. Increasing its tolerance
would conceal the unsupported placement, and simply repeating the same settled
reference would not address the measured tracking offset. Diagnosis records are
preserved in `experiments/2026-11-26_place_tracking_fix/diagnosis.json`.

## Bounded tracking and the rejected first candidate

New `control/tracked_place.py` adds a final-lowering Cartesian tracking bias:

- Activate only within 15 mm of the destination and after the current reference
  catches its requested goal within 1 mm.
- Integrate measured goal-minus-pinch error with gain 0.2 per control period.
- Clamp each Cartesian bias component to +/-6 mm; reset at every lowering phase.
- Stop integrating once the existing stable-support readiness condition is met.
- Pass all proposals through the original rate, IK, table-clearance and contact
  checks. Other phases have unchanged control logic.

This repairs the recorded last timeout and passes **120/120** new cases at seeds
90000–90009. However, it regresses **eight previously successful cases** in a
120-condition recorded recheck, completing only 112/120. It keeps integrating
downward after table support when XY is still slightly outside readiness, until
the transfer/opening clearance guard rejects the proposed motion. These are
guarded clearance rejections, not observed robot/table collisions.

That candidate is **rejected despite its perfect fresh score**. Both its fresh
outcomes (`2026-11-26_place_tracking_validation/`) and recorded failures
(`2026-11-26_place_tracking_fix/recorded/`) remain intact, with a combined rejection
record. No result was overwritten to make the change look successful.

## Final contact-aware correction

New `control/contact_tracked_place.py` retains the bounded airborne tracking bias
but changes behavior when the cube is already supported:

1. Stop vertical integration immediately on cube/table contact, even if the
   horizontal readiness condition has not passed.
2. Hold the current vertical reference while supported and continue only the
   caught-up, bounded horizontal correction.
3. If support is absent, use the same bounded airborne correction/search.
4. Open only after the unchanged five-sample readiness condition authorizes it.

The original 3 mm touchdown search remains the nominal cube-target adjustment.
The separate actuator-tracking bias can shift an IK reference by up to 6 mm per
axis; do not describe that as a maximum 3 mm total command displacement. Actual
motion still obeys joint-rate/clearance checks and measured table support. This
is not a calibrated hardware contact-force controller.

No readiness or safety criterion changes: five supported/low-speed samples,
XY <=5 mm before opening, height <=5 mm, speed <=0.01 m/s, final released-rest
XY <=10 mm, >=40 mm hand retreat, 4 mm nominal robot/table clearance. The lower
budget remains 150 periods. The original 200-step acquisition and additional
transfer-phase budgets remain unchanged.

## Recorded recheck and independent validation

The final candidate was frozen after bring-up on two recorded cases: the last
timeout above and the new 15 g / -30-degree / seed 88000 clearance regression.
No final validation outcomes were used to tune it.

| Check | Previous result | Final candidate |
|---|---:|---:|
| Original supported-transfer recorded set, seeds 88000–88004 | 60/60 | 60/60 |
| Last supported-transfer validation, seeds 89000–89004 | 59/60 | **60/60** |
| First tracking candidate's new set, seeds 90000–90009 | 120/120 | 120/120 |
| Total recorded conditions | 239/240 | **240/240** |
| Independent new seeds 91000–91009 | Not previously tested | **120/120** |

Each new seed repeats four masses (5/10/15/20 g) and three yaw offsets
(-30/0/+30 degrees), in fixed nominal calibrated arm effects. Every final
mass/yaw cell completes 10/10. All 120 fresh acquisitions pass; no final candidate
episodes stop at a timeout, IK/clearance rejection or grasp loss.

This is **ten new placement centers**, not 120 independent placements. The 240
recorded conditions reuse earlier placement sets and are regression evidence,
not additional independent validation. Finite 100% success does not establish
zero underlying failure probability, arbitrary-goal reachability or physical-
robot reliability. Destination geometry remains the nearby 12-degree same-radius
table move from the first complete-task prototype.

Fresh final XY placement error averages **4.507 mm**, maximum **6.825 mm**, after
release, retreat and the final resting observation. All completed cubes are
supported, not grasped and below the original final speed/height thresholds.
Maximum fresh task duration is 484 periods, within the existing full-sequence
budget; there was no horizon extension to recover failures.

The repaired original failure completes in **459 periods (15.147 seconds)**:
pickup 200, carry 36, lower 70, release 60, retreat 33 and settle 60. Final cube
center is (0.024125, -0.080300, 0.487392) m against destination
(0.018996, -0.077785, 0.487500) m: **5.712 mm XY error**, with the cube released
and stationary. Its trace and phase images are under `repaired_case/`.

## Selected physics-step audit

Four selected complete sequences audit every 1 ms physics step: the last 5 g
timeout, the 15 g clearance regression, and two fresh 5/20 g conditions.
All four complete, with **zero robot/table contacts across 59,037 physics steps**
and no warnings. Before opening, each passes the original support-readiness gate
and has positive measured-in-simulation cube/table normal force, about 0.228–0.810 N.
These solver forces are not hardware measurements or prescribed contact limits.

This complements control-boundary counters on the full matrix; it does not
certify every case at physics-step resolution. The audit wraps `mj_step` only
inside its own process and restores it even after exceptions, without changing
state or commands.

## Decision, code and tests

`combined_assessment.json` accepts the final experimental controller: all prior
conditions retained, all fresh conditions pass the frozen per-cell criterion,
zero new failures and zero sampled contacts/warnings. The rejected unconditional
integral remains clearly marked as rejected. No automatic training starts.

The next development decision can now move beyond the lowering bug: preserve
this exact full sequence as a reproducible baseline and define the intended
complete-task learning environment/objective if RL is required. The demonstrated
success is scripted control with privileged cube pose/contact state, not a
newly trained policy. Old reaching/pickup policies and defaults remain separate.

New runtime modules: `tracked_place.py`, `contact_tracked_place.py`.
The existing `carry_place_check.py` runner adds explicit, mutually exclusive
`--tracked` and `--contact-tracked` selections; default and `--supported` behavior
remain available. No existing runtime controller/model was edited.

Tests were added first and failed before implementation. **444 pass, one optional
LeRobot test skips.** Tests cover the last failure, exact unchanged pickup/carry
physical state, stopped integration once support is ready, the introduced
clearance-regression repair, bounded correction and frozen vertical bias during
supported horizontal correction. Original support and clearance tests still pass.

Artifacts include both rejected and accepted outcomes, frozen protocols, full
recorded rechecks, original diagnosis, combined assessments, repaired-case images,
physics audit, comparison plot, source/software provenance and input hashes.
Protected model/calibration/runtime/policy hashes match around all frozen runs.
No dependency sync or training ran; user-local CUDA edits and personal viewer
directories remain untouched.

```powershell
# Watch the formerly failing case complete; use a NEW output directory
uv run --locked python -m scripts.sim2real.analyze.carry_place_check --contact-tracked --replay --seed 89000 --mass-kg 0.005 --yaw-offset-deg 30 --viewer --out experiments/place_fixed_viewer_1

# Repeat the 120-case independent protocol, without training
uv run --locked python -m scripts.sim2real.analyze.carry_place_check --contact-tracked --out experiments/contact_place_repeat
```

The viewer plays recorded physical poses after a headless run. Output directories
must be new. The local baseline run configuration and old checkpoint companions
are required for verification; this runner does not load a neural policy.

# Raise-before-traverse acquisition experiment — 2026-11-26

The raise-only path is a **partial improvement**, not a complete acquisition fix.
On fresh placements with arm effects, lifts improve from **195/200 to 198/200**,
and observed initial-approach contacts fall from 28/200 episodes to zero. However,
the cube still moves too much before closing, and the known 5 g failures remain.
The full acquisition gate fails. Do not promote defaults or retrain yet.

Measured alignment checks and bounded correction were also tested. Checking alone
aborts many grasps; adding correction repairs known cases but causes table contacts
on new tuning placements. Those correction variants are rejected.

## Scope and implementation

Base commit: `66ee158`, integration/calibrated-pick-lift. New opt-in controller
`RaisedApproach`, `ApproachConfig`, and `AcquisitionGuard`; separate headless
ablation/replay runner. Original working controller, model, physical gains,
collision settings, calibration, saved policies, reward and training presets remain
unchanged. No policy was trained or executed on the new sequence.

The raise stage uses position IK at the initially sensed pinch XY, with free
orientation. Its height goal is the greater of initial pinch Z +50 mm or cube
center Z +80 mm. Traverse begins when measured height is within 25 mm of that
goal (at the current home pose, at least about 25 mm measured rise). Then execute
the original aligned pre-grasp target. The original phase clock remains:
40 approach, 30 descend, 30 close, 100 lift; 200 steps / 6.6 s total.

The initial 8 mm raise-goal tolerance stalled under arm effects. It was adjusted
using known diagnostic placements before independent validation; no gains changed.

The optional pre-close check requires sensed XY error <=8 mm, Z error <=8 mm,
and approach tilt <=8 degrees for three consecutive measurements. Errors are
relative to the existing cube-center-minus-4-mm pinch target, not the pure IK
residual. While waiting, hold the jaw open. Abort open if raising or alignment
cannot complete in the existing phase windows. The new guard prevents the legacy
lift guard from overriding an acquisition abort with a closing command.

The experimental correction variant updates a Cartesian target bias every three
waiting steps with gain 0.5, limited to +/-15 mm **per axis**, then solves existing
IK with existing gravity feedforward. It does not change actuator gains. This
variant has no clearance-aware trajectory guarantee and is rejected by the tests
below; it is not enabled in the selected raise-only configuration.

## Protocol and ablations

Five variants: unchanged baseline, raise only, alignment correction only, raise
plus alignment check, and raise plus alignment correction. Known regression seeds
68000/68006/69005 at four masses and two dynamics modes: 24 episodes per variant.
Fresh tuning seeds 70000–70009: 80 episodes per variant. Total tuning/regression:
**520 episodes**.

Freeze the candidate before fresh held-out seeds 71000–71049, four masses, both
dynamics modes, baseline versus candidate: **800 episodes**. Placements are reused
across masses and modes; they are not 800 independent random locations.

Predeclared gate: >=95% final success in every mass/mode; zero observed robot/table
or initial-approach cube-contact episodes; at most 5% of episodes with >1 mm
horizontal cube movement before closing; no warnings. Known regressions must lift.
An open abort counts as a failed lift, not as a successful grasp.

| Variant | Known successes /24 | Tuning successes /80 | Tuning table-contact episodes |
|---|---:|---:|---:|
| Baseline | 21 | 80 | 0 |
| Raise only | 22 | 80 | 0 |
| Correction only | 24 | 80 | 10 |
| Raise + alignment check | 16 | 44 | 0 |
| Raise + alignment correction | 24 | 80 | 10 |

No variant passes the full tuning gate, including the pre-close movement limit.
`raise_only` is frozen as the best collision-free exploratory comparison, explicitly
with `passed_tuning_gate=false`. It is not described as a validated full fix.
The known 15 g failure at seed 68006 is repaired; 5 g failures at 68006 and 69005
remain. Correction repairing these known cases does not justify overlooking its
fresh tuning table contacts.

## Held-out comparison

Final successes out of 50 placements per mass:

| Mass | Baseline, stiffness | Raise only, stiffness | Baseline, arm effects | Raise only, arm effects |
|---|---:|---:|---:|---:|
| 5 g | 50 | 50 | 47 | 48 |
| 10 g | 50 | 50 | 49 | 50 |
| 15 g | 50 | 50 | 49 | 50 |
| 20 g | 50 | 50 | 50 | 50 |
| Total | **200/200** | **200/200** | **195/200** | **198/200** |

For arm-effect cases:

| Acquisition metric | Baseline | Raise only |
|---|---:|---:|
| Initial-approach contact episodes | 28/200 | **0/200** |
| Robot/table-contact episodes | 0/200 | 0/200 |
| Cube movement >1 mm before closing | 105/200 | **77/200** |
| Median pre-close XY movement | 1.034 mm | 0.825 mm |
| Maximum pre-close XY movement | 27.105 mm | 2.226 mm |
| Episodes losing achieved success | 2/200 | 0/200 |

Three matched outcomes improve: 5 g/71043 and 10/15 g/71012. No matched final
success regresses. This is a small descriptive improvement from one deterministic
controller comparison, not a population reliability guarantee.

The worst mass/mode has 84% of episodes above the 1 mm pre-close movement limit,
far above the frozen 5% limit. Therefore `passed_heldout_gate=false`, despite the
improved lift count and absence of table contact. Do not loosen the frozen limit
after seeing the outcome.

Both stiffness-only controllers have no observed approach/table contacts and
negligible pre-close XY movement. Remaining movement under arm effects occurs
during descent/seating, after the new raise stage has eliminated initial approach
contact. A single joint-space descent target still permits an unfavorable physical
path and tracking error. Closing-stage position correction alone can drive the
fingers into the table even when ideal IK converges.

## Validation and evidence

- **397 tests passed, one optional LeRobot test skipped.** New tests first failed
  before implementation, then passed. They cover measured closure thresholds,
  bounded bias, disabled-option equivalence with old actions and keeping the jaw
  open after an acquisition abort despite the lift guard.
- No MuJoCo warnings in the formal tuning or held-out runs.
- Physics-step audit: 12 episodes, baseline/raise-only, 5/15 g, full arm effects,
  seeds 68006/69005/71000, **79,200 physics steps**. Initial-approach cube contact
  occurs on 368 baseline substeps and zero raise-only substeps. Neither has table
  contact in the audit. This subset does not certify all substeps of the full run.
- Saved phase PNGs/traces show repaired 15 g/68006 and remaining 5 g/68006. The
  latter briefly grasps at step 83 and loses contact at 87; the former reaches
  success at 116.
- Model/calibration/original-checkpoint hashes match; v2 files match the preceding
  frozen holding-evaluation manifest. Existing controllers and weights are intact.

Evidence directory: `experiments/2026-11-26_raised_approach/` contains protocol,
all tuning and held-out episodes, frozen selection, assessment, source provenance,
substep audit and the two replays. Rejected variants are retained as evidence.

## Commands and next step

Repeat into a new directory:

```powershell
uv run --locked python -m scripts.sim2real.analyze.raised_approach_check --out experiments/raised_approach_repeat
```

Watch the selected **raise-only exploratory** configuration:

```powershell
uv run --locked python -m scripts.sim2real.analyze.raised_approach_check --replay experiments/2026-11-26_raised_approach/frozen_selection.json --mass-kg 0.015 --placement-seed 68006 --arm-effects --viewer --out experiments/raised_approach_viewer
```

Use 5 g to see the remaining failure; omit `--viewer` for headless phase images.
Each output directory must be new. Existing saved RL policies are not automatically
compatible with this changed acquisition sequence and were not combined with it.

Next test a clearance-aware Cartesian descent from a measured aligned pre-grasp,
with attention to the stationary finger's swept path. Validate actual cube motion
and table clearance before trying more closing-stage correction. Preserve the
old baseline and keep the new path opt-in; no full training is warranted yet.

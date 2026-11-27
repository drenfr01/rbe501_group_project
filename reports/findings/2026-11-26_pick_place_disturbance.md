# Frozen full-task policy: small carry-push robustness comparison

**Both controllers pass the tested pushes; useful added RL robustness is not
established.** Each completes 320/320 pushed placements plus 40/40 no-force
controls, with no sampled grasp losses, robot/table contacts or MuJoCo warnings.
The task/model/weights are unchanged and no training occurred.

## Results and decision

| Outcome | Scripted | Frozen full-task SAC |
|---|---:|---:|
| Eligible initial carrying holds | 360/360 | 360/360 |
| Complete placements after nonzero push | **320/320** | **320/320** |
| No-force controls | 40/40 | 40/40 |
| Sampled unexpected transport grasp losses | 0 | 0 |
| Robot/table contact episodes | 0 | 0 |
| MuJoCo warning episodes | 0 | 0 |
| Mean final XY error under pushes | 4.829 mm | 4.670 mm |
| Mean final XY error without push | 5.002 mm | 4.723 mm |

Every magnitude/direction condition completes 40/40 per controller, across the
four masses and ten placements. All actual pulse windows are exactly three
carry periods / 99 ms, with no ineligible or phase-mismatched trials. All 320
disturbed pairs have full observed path windows; no failure or window is censored.

The paired retention difference is zero. Its empirical bootstrap interval [0, 0]
reflects identical observed success, not guaranteed population equivalence or a
zero failure probability. These small forces do not find the grip-failure boundary.

Raw final error is slightly smaller for SAC, but part of that difference already
exists in the no-push control. After subtracting each controller's matched
no-force error, learned-minus-script difference is **+0.120 mm**, 95% placement-
cluster bootstrap interval **[-0.121, +0.365] mm**. This interval includes both
improvement and worsening. It does not establish a difference, and misses the
predeclared 0.5 mm precision-benefit threshold.

Secondary descriptive movement results:

| Metric relative to own matched unforced path | Scripted | SAC |
|---|---:|---:|
| Mean cube path RMS deviation | 0.210 mm | 0.255 mm |
| Mean peak grip-relative deviation | 0.054 mm | 0.052 mm |
| Maximum grip-relative deviation | 0.184 mm | 0.134 mm |
| Maximum world cube path deviation | 2.188 mm | 2.251 mm |
| Stable-return confirmation | 320/320 | 320/320 |

Both stay within the 1 mm grip-relative tolerance throughout these windows, so
all stable-return confirmations occur at the earliest possible five post-pulse
samples (165 ms). This is evidence of maintained holding, not faster recovery
from a dropped cube. No controller is shown to recover from grasp loss here.

The policy was active rather than silently replaced: it applies learned residuals
on 32,635 pushed-trial control periods (36,827 including controls). The scripted
baseline has no learned residuals. Full reward, observation layout, scripted
acquisition/release guards and normalizer statistics remain frozen.

**Decision:** retain the validated full-task policy and scripted reference as
working simulation artifacts, but do not launch a longer training campaign from
this result alone. The complete nominal task and this modest-push benchmark are
already saturated by the script. If the class project needs evidence of an RL
advantage, define a demonstrated task/model gap and measurable learning objective
before retraining. Further evaluation should target a documented transfer gap.

This evaluates nominal cube yaw, fixed nominal arm effects, one trained seed and
one uncalibrated contact/gripper model. It does not establish robustness to model
uncertainty, arbitrary destinations, hardware perception or physical disturbances.
The reported contacts are sampled at control boundaries, not every physics step.

Evidence includes all 720 outcomes/windows, 320 paired comparisons, assessment,
grouped summary, figure/source, frozen protocol, source/software provenance and
before/after checksums. Exact policy, normalizer, run configuration, runtime,
model/assets and calibration fingerprints match. Previous results and weights
are not overwritten.

## Protocol

Compare the saved `pick_place_residual_v1` SAC policy against its matching
zero-residual scripted controller, without training or changing runtime code,
model, gains, rewards, normalizer statistics, task horizon or success criteria.
Base integration commit: `f62ad58`.

The protocol is recorded before rollouts under
`experiments/2026-11-26_pick_place_disturbance/protocol.json`:

- Ten new placement seeds **94000–94009**, four fixed masses 5/10/15/20 g,
  nominal cube yaw and fixed nominal calibrated arm effects.
- No-force controls plus **0.1 N and 0.2 N** pulses in +/-world X and Y.
- World-space force at the cube COM, no added torque, for exactly three 33 ms
  periods (**99 ms**), at zero-based task steps 200–202: t=6.600–6.699 s.
  This is the first carrying motion after the unchanged 200-step pickup.
- Force requires the actual carry phase and the preceding five samples to show
  a retained two-pad grasp, >=4 cm lift and no table support. Ineligible trials
  remain in overall complete-task outcomes.
- The force schedule is identical between controllers, not shifted to each
  controller's later progress. Actual pulse duration and physical phase are
  recorded. The existing wrench is restored after every period, even on errors.
- **720 episodes total:** 360 per controller, including 40 controls and
  320 nonzero-push cases per controller. Conditions repeat ten placement centers;
  masses/directions are not independent placement samples.

Primary outcome is complete placement after release/retreat, using common
pre-eligible pairs. A failure during a partial pulse remains a failed disturbance
trial; it is not censored for preventing the remaining pulse delivery. Final
errors include failures, rather than conditioning on successful placement only.

Secondary placement comparison subtracts each controller's matching no-force
final XY error, then computes learned-minus-script difference. Bootstrap the
placement-cluster means with 10,000 resamples, seed 0. Positive differences favor
the script; negative favor RL. The frozen useful-benefit rule requires a positive
retention-difference interval or an adjusted error improvement interval entirely
beyond **0.5 mm**, with no additional paired final failures, no-force controls
passing and no policy table contacts/warnings. The 0.5 mm threshold is a
preselected simulation benchmark, not a validated hardware tolerance.

## Movement and recovery measurements

Each controller has its own matched no-force trajectory. Record true physical
cube/pinch positions during the three pulse periods and 30 following periods.
Subtract the controller's no-force positions at the same absolute periods;
this distinguishes a push-associated path deviation from the intended carry
and lowering motion. Path RMS is not static holding error or IK target error.

Grip-relative deviation compares `(cube - pinch)` with the same controller's
unforced path. Recovery requires five consecutive post-pulse samples within
1 mm of that nominal grip-relative path while both trials retain their grasp.
Earliest confirmation is **165 ms** after force removal; that is a confirmation
window, not an estimate of physical settling time after a lost grasp. Released
or dropped objects cannot count as recovered simply because they stop moving.

If an episode terminates before the path window ends, its unobserved motion is
not padded or invented. The path diagnostic is unavailable for that window,
while its complete-task failure remains in the primary comparison.

## Validation and reproduction

Tests were introduced before implementation. **457 tests pass, one optional
LeRobot test skips.** New tests check exact three-period pulse timing, carry
eligibility, grasped stable recovery, retention of a failure during a partial
pulse, and exact physical qpos/qvel equality of a no-force instrumented rollout
with the original full-task evaluation. Previously tested wrench restoration
and isolated-cube impulse checks remain in the suite.

New evaluation code is `scripts/sim2real/analyze/pick_place_disturbance.py`, with
typed immutable protocol/pulse/outcome records. No existing runtime or training
code was edited. Full-task model/statistics/configuration and source/model/assets/
calibration hashes are checked before and after evaluation. Old policies and
results are preserved; no optimizer or observation-statistics updates occur.

```powershell
uv run --locked python -m scripts.sim2real.analyze.pick_place_disturbance --model-dir checkpoints/pick_place_v1_pilot --out experiments/full_task_disturbance_repeat
```

Use a new output directory. The local complete-task model, matching normalizer,
run configuration and exact runtime/model assets must be available. This evaluates
the new integration-branch full-task residual policy, not main's original reaching
or pickup policies. No long or short training job is launched.

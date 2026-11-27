# Guard, clearance, and manipulation arm-effects validation (2026-11-26)

## Decision

**Do not retrain or promote this candidate to defaults yet.** The requested
guard and clearance experiments were implemented and tested, a candidate was
frozen before independent evaluation, and the old policies were reevaluated.
The result is negative: acquisition/retention and table contact remain
unresolved. Added arm effects are software-validated simulation assumptions,
not a validated model of loaded physical manipulation.

## 1. Grasp-gated lifting

`GuardedLift` filters the final executed action below learned residual control.
At the scheduled lift phase it requires five consecutive samples of two-pad
contact. While waiting, or after contact loss, arm commands hold and the jaw is
commanded closed. Loss after lift release latches an abort for that episode;
closing-phase contact changes do not prematurely latch it. No retry/regrasp
strategy is introduced. A queued delayed arm command may still take effect.

The gate uses privileged simulator contact information, not a real grasp
sensor or measured grasp force/quality. It is not a physical safety controller.

The previous 5 g drop case (random-mass policy, seed 20008) still drops. The
guard aborts at step 125, the control step after the recorded loss, and arm
commands remain fixed thereafter. It prevents continued command motion, not
the already-occurring fall. No success-rate gain was found in held-out tests.

## 2. Clearance tuning

Use scripted control, measured stiffness, four masses (5/10/15/20 g), and ten
matched placements per mass (seeds 40000–40009): 240 tuning episodes.
Change only gating and the grasp-height offset; contact geometry/friction,
actuators, phase durations, and episode length are unchanged.

| Setting | Success / 40 | Episodes with robot/table contact |
|---|---|---|
| Original schedule | 39 | 39 |
| Grasp gate, 0 mm offset | 39 | 39 |
| Grasp gate, +2 mm | 37 | 32 |
| Grasp gate, +4 mm | 33 | 27 |
| Grasp gate, +6 mm | 31 | 8 |
| Grasp gate, +8 mm | 27 | 0 |

Raising the waypoint trades table contact for missed grasps. It is not a
validated clearance fix. The predeclared selection first requires 95% success
where available, then minimizes contact episodes, then breaks ties by success
and smaller offset. It selects the gated **0 mm** setting. This is an
experimental choice, not approval of its frequent table contact.

## 3. Which arm dynamics can be exercised

`GraspServoBackend` composes the existing reaching wrappers around the physical
grasp backend while preserving gripper commands, object placement/mass, and
grasp-state access. Its sensed pinch position is recomputed from sensed joints;
cube pose and contacts remain simulator truth. Fixed-effect resets use a
separate RNG so corresponding placement seeds remain matched.

The explicit arm-effects condition uses the measured stiffness JSON plus the
nominal deployment values of `residual_ik_sim2real_smooth`:

- fitted arm stiction half-bands/load slopes and transition ratio;
- relative arm-goal clipping;
- one control-step arm-command delay (33 ms);
- encoder quantization (2*pi/4096 rad) and corresponding sensed velocity/FK.

Not included/validated: moving-gripper dynamics/command delay, loaded-grasp
stiction fitting, real joint-window feasibility of this manipulation workspace,
physical pad friction/compliance, or object perception. Pan and wrist-roll
stiffness are assumed values in the source fit. Transferring a reaching fit
does not establish loaded-manipulation fidelity.

Regression checks cover disabled-effect identity, matching cube placements,
gripper/object state preservation, explicit calibration requirements, delayed
arm commands, quantized sensing, and filtering of final actions. These validate
the software interface, not the physical model's accuracy.

## 4. Frozen candidate and independent old-policy evaluation

`frozen_candidate.json` was written after tuning and before evaluating the
policies on 20 separate placements per mass (50000–50019). Candidate settings:
measured stiffness, unchanged 0 mm grasp offset, five-contact-sample guard,
unchanged baseline task and contacts. The additional arm-effects condition is
a separately labeled stress test. No held-out result was used to retune it.

720 held-out episodes, 80 per cell:

| Dynamics/controller setup | Scripted | Fixed-10-g SAC | Random-5–20-g SAC |
|---|---|---|---|
| Measured stiffness, original schedule | 74/80 | 67/80 | 75/80 |
| Measured stiffness, grasp-gated | 74/80 | 67/80 | 75/80 |
| Measured stiffness + arm effects, grasp-gated | 66/80 | 52/80 | 71/80 |

Table-contact episodes respectively:

| Setup | Scripted | Fixed-10-g SAC | Random-5–20-g SAC |
|---|---|---|---|
| Original schedule | 79/80 | 80/80 | 80/80 |
| Grasp-gated | 79/80 | 75/80 | 80/80 |
| Arm effects + guard | 0/80 | 74/80 | 76/80 |

All 960 tuning/held-out episodes remained finite with no MuJoCo warnings.
Contact is sampled at control boundaries, not every physics substep. Only
seed-0 learned checkpoints were available. These small, correlated placement
sets do not establish population success rates or statistical significance.

Measured stiffness alone versus arm effects changes performance substantially;
that is evidence of policy sensitivity, not proof that either model matches
the real robot. Neither a grasp gate nor a larger height offset solves the
underlying contact-alignment/retention problem by itself.

## Frozen artifacts, replay, and next work

New records: `experiments/2026-11-26_guarded_pick_lift_validation/` contains
`tuning.json`, `frozen_candidate.json`, `heldout.json`, `validation_manifest.json`,
`assessment.json`, and the guarded drop replay. The manifest includes the
executed script hashes and exact effects/configuration values. Input model,
calibration, policy, normalizer, and run-info hashes were unchanged. Later
API hardening rejects a missing calibration file and supports replay rendering;
it does not change the valid numerical conditions recorded in the sweep.
Eighteen matched 10 g episodes across all three setups/controllers were replayed
after this hardening and reproduced success flags and final lifts within
1 micrometer. A separate arm-effects replay with offscreen frames also passed.
The final automated suite reports **370 passed, 1 skipped** (optional LeRobot
configuration check).

The freeze is a reproducible **experimental setup**, not a production/default
or hardware approval. `assessment.json` explicitly records
`ready_for_training: false` and `promote_to_defaults: false`.

Next isolate grasp alignment and retention (including jaw contact placement and
lift orientation/trajectory), using scripted control first. A quality/contact
criterion stronger than binary two-pad touch may be needed. Validate the
gripper/contact assumptions against hardware before claiming a fully calibrated
manipulation model or spending a long run on retraining.

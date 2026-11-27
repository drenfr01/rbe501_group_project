# Phase-limited lift residual pilot — 2026-11-26

The second short pilot **passes the frozen simulation gate** and matches the
scripted reference: **78/80 successful lifts, zero sampled table contacts, zero
losses after achieving success**. The failed v1 policy on these same new cases
gets 46/80, with table contacts in 66/80 and 28 losses after achieving success.

This restores baseline reliability; it does **not** demonstrate that RL improves
on the script. Keep the script as the reference and leave defaults unchanged.
No full training, reaching-policy integration or hardware execution was started.

## 1. Preserve the reference

The validated outer-wrist script, source MJCF, physical gains, task reward, cube
geometry, calibration, original teammate policies and failed v1 policy remain
unchanged. A new opt-in wrapper and serialized v2 preset restrict only the new
pick/lift learner's correction authority. A zero-residual regression matches the
legacy scripted stack for all 200 steps of a complete episode.

## 2. Diagnose the failed policy by phase

Frozen v1 weights, known diagnostic placements 66000–66004, four masses, five
placements per mass: 20 episodes per row, 140 total. Disable residuals outside
each named phase without retraining. These placements were already used for v1
evaluation; they are diagnostic, not independent validation.

| Applied v1 residual | Successful lifts / 20 | Table-contact episodes / 20 |
|---|---:|---:|
| All phases | 4 | 11 |
| Approach only | 10 | 0 |
| Descend only | 20 | 10 |
| Close only | 19 | 0 |
| Lift only | 11 | 4 |
| None (script) | 19 | 0 |
| Lift only, one-tenth amplitude | 19 | 0 |

The all-phase case has sampled contacts in descent, close and lift; the lift-only
case loses achieved success eight times. Descent-only corrections can increase
grasp success while introducing unwanted contacts. These interventions change
downstream states, so phase effects cannot be added as independent causal terms.

Evidence: `experiments/2026-11-26_residual_phase_diagnostic/phase_ablation.json`.

## 3. Restrict authority and check proposals

New `calibrated_outer_wrist_lift_only_v2` preset:

- Zero learned correction during approach, descent and closing. Gripper correction
  is always zero. Arm corrections require the lift phase, current two-pad contact,
  and a released, non-aborted grasp gate.
- Residual scale 0.05 rather than 0.5. At the existing arm action scale, the raw
  learned correction is at most 0.0035 rad per arm joint per control period.
- Project proposed commands into +/-0.02 rad of a separate rate-limited scripted
  command reference. The reference advances even when corrections are rejected.
- Use private MuJoCo data for nominal command-pose checks. Require >=2 mm distance
  between collidable robot geometry and the table top; disabled visual meshes are
  excluded. Once success has occurred, reject proposals more than 0.5 mm below
  the current scripted proposal's predicted pinch height.
- Reject failed checks by falling back to the existing scripted action. Keep the
  existing contact guard beneath the final executed action.

These are **nominal kinematic proposal checks**, not dynamics predictions, swept
path checks, or hardware safety guarantees. They do not model tracking sag, delay,
contact deformation or cube-to-pinch motion. Fallback uses the validated script;
the checker does not certify an already-unsafe scripted fallback. The command
budget projection can also adjust a proposal independently of the raw learner's
action scale.

New modules: `rl/lift_residual_limits.py` and `rl/limited_lift_training.py`.
The dedicated repository records all limits and the base environment snapshot,
refuses existing run directories, and uses a dedicated v2 policy loader. The
legacy loader and v1 checkpoint continue to work.

## 4. Fresh short pilot and independent evaluation

Freeze the envelope and criteria before looking at held-out placements. New SAC
seed 0, **10,000 transitions / 50 complete episodes**, CUDA on RTX 5070. Keep the
previous SAC hyperparameters, reward, 5–20 g training mass distribution and arm
effects. Start from new weights and statistics, not v1 or a reaching checkpoint.

Evaluate saved/reloaded deterministic policies at fresh seeds **68000–68019**,
four fixed masses. The script, frozen v1 and v2 use matched placements and masses.
This is 20 independent placement seeds reused across masses, not 80 independent
random locations per controller.

Each cell is successful final lifts / table-contact episodes, out of 20:

| Mass | Script | Failed v1, matched cases | New constrained v2 |
|---|---:|---:|---:|
| 5 g | 19 / 0 | 8 / 17 | 19 / 0 |
| 10 g | 20 / 0 | 12 / 17 | 20 / 0 |
| 15 g | 19 / 0 | 13 / 17 | 19 / 0 |
| 20 g | 20 / 0 | 13 / 15 | 20 / 0 |
| Total / 80 | **78 / 0** | **46 / 66** | **78 / 0** |

Gate: >=95% final success and <=5% table-contact episodes at every mass, no MuJoCo
warnings, plus <=5% losses after achieving success at every mass. V2 passes all
observed thresholds. Script and v2 have zero lost-success episodes; v1 has 28.
The earlier v1 result of 29/80 used different seeds (66000–66019); use the matched
46/80 comparison here. No MuJoCo warnings in any of these evaluations.

Both script and v2 fail at seed 68006 for 5 g and 15 g. At 5 g, pad contact appears
at step 84 and is lost at 88, before lift release. V2 applies no corrections in
that episode. Protecting acquisition intentionally leaves this existing weakness
for a separate acquisition experiment; longer lift-only training cannot directly
control that phase.

### Is the learner actually active?

Yes. Held-out evaluation contains **7,722 active correction steps**, all accepted
by the nominal pose checks, with 863 command-budget clips. Thus this result is
not obtained by always disabling the policy. No clearance/holding proposals are
rejected in deterministic evaluation; their contribution cannot be claimed from
these outcomes alone.

During exploratory training: **49/50 successful episodes**, zero sampled table
contact episodes, no MuJoCo warnings; 419 holding proposals rejected and 289
budget clips. One training episode triggers the grasp-abort guard. The proposal
statistics are collected before the inner contact guard; they are not physical
actuator motion measurements. Training statistics are not final-policy validation.

## Additional verification

- **388 tests passed, one optional LeRobot test skipped.** New tests first failed
  before implementation; coverage includes phase/jaw masks, prediction thresholds,
  actual fallback on a predicted height violation, immutable physical state during
  prediction, zero-residual equivalence, serialization and abort-latch behavior.
- SB3 environment checker and CUDA arithmetic passed; SAC reports device `cuda`.
- A final abort-latch eligibility correction prevents residual proposals after an
  abort even if contact reappears. The existing inner guard already held the arm.
  All 80 saved-policy evaluations were rerun after this correction and match the
  original v2 outcomes and counters exactly. Original training logs are retained.
- Physics-step contact audit: script/v2, 5/20 g, seeds 68000/68006, eight episodes,
  **52,800 physics steps**, zero robot/table-contact substeps. Outcomes agree with
  the main evaluation. This subset audit does not certify every substep of all
  evaluation episodes.
- Successful 10 g replay and shared 5 g acquisition-failure replay include full
  traces and phase images. Protected model/calibration/old-policy hashes match.

## Artifacts and commands

`experiments/2026-11-26_limited_lift_pilot/` preserves the frozen protocol, resolved
preset, provenance, versions, matched script/v1/v2 episodes, intervention counters,
50 training reward records, final assessment, hash inventory, replays and audit.
Binary v2 model, normalizer, monitor and replay buffer remain **local** under
`checkpoints/calibrated_lift_only_v2_pilot/`; they are excluded by the existing
checkpoint ignore rule and are not uploaded with the code/findings.

Repeat only into new directories:

```powershell
uv run --locked python -m scripts.sim2real.analyze.limited_lift_pilot --out experiments/lift_only_v2_repeat --model-dir checkpoints/lift_only_v2_repeat
```

Watch the saved v2 policy with its matching loader:

```powershell
uv run --locked python -m scripts.sim2real.analyze.limited_lift_pilot --watch --model-dir checkpoints/calibrated_lift_only_v2_pilot --mass-kg 0.01 --placement-seed 68000 --viewer --out experiments/lift_only_v2_viewer
```

Replay always saves phase PNGs. Use 5 g / seed 68006 to inspect the shared failure.
Do not load v2 through the legacy v1 replay command, which lacks the new envelope.

Next: retain the script as the best-established baseline and v2 as an opt-in stable
learner candidate. Define a measurable benefit (such as holding precision under
specified perturbations) and validate that benefit across additional training seeds
before a full run. Treat acquisition correction separately; it needs an explicit
new test and should not silently restore unrestricted descent corrections.

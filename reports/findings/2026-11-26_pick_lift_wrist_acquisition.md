# Wrist branch and grasp acquisition — 2026-11-26

The opt-in `outer_wrist` controller passes the fresh simulation gate: **200/200**
final successes with calibrated stiffness alone and **199/200** with nominal arm
effects. The previous motion candidate achieves 200/200 and 192/200 respectively
on exactly the same placements. No robot/table contacts were observed at control
boundaries in either controller's 800 comparison episodes. No MuJoCo warnings.

This is a scripted simulation candidate, not a retrained policy or a hardware
validation. Defaults, gains, source MJCF, saved weights and calibration are unchanged.

## Diagnosis and rejected changes

At the previous 10 g failure placement 61010, step 70 measured pinch error is
approximately 17.1 mm in X and 20.0 mm in Z. The ideal IK solution remains near
its target; actual arm configuration and finger/cube interaction matter. The
new seed reduces those errors to about 7.2 mm and 6.7 mm before closing, and
the replay finishes with 5.19 cm lift.

Early in-memory probes of bounded tracking feedback introduced table contacts or
failed acquisition. Longer dwell and stiction feedforward probes were insufficient.
These provisional controls were removed, not made part of the final controller.

Trying a wrist seed shifted by approximately 180 degrees allows the solver to
find a different configuration and reverses the TCP X direction. This changes
which side the stationary finger approaches. It is an IK seed, not an enforced
180-degree Cartesian orientation; existing IK mathematics and joint limits remain.
Both old diagnostic failures lift successfully with this seed at 10 g.

However, applying that seed everywhere is worse at 5 g. In the new ten-placement
tuning batch 62000–62009, the original candidate achieves 100% with both dynamics
modes and all masses. Constant negative-pi wrist seed gives 60% at 5 g with arm
effects; positive-pi gives 40% (and 80% at the other masses). Both give 100% with
stiffness alone. These are rejected global changes, preserved in
`experiments/2026-11-26_pick_lift_wrist_branch/tuning.json`.

That ablation selected the original candidate. Its subsequent 63000–63049 batch
still gives only 94/94/92/92% success at 5/10/15/20 g with arm effects, so a fresh
batch confirms the remaining weakness. The runner evaluated the selected original
candidate twice because it was also the reference; those duplicate rows are
deterministic repeats, **not additional independent evidence**.

## Final opt-in change

Retain the previous yaw alignment, bounded gravity feedforward and -4 mm waypoint.
At approach entry, compute cube radius around the shoulder-pan axis from the same
object position already present in the observation. Use a negative-pi wrist seed
only at radius >=210 mm, the outer 10 mm of the configured 140–220 mm annulus.
Keep the original seed inside that radius. The two diagnostic outer failures
are approximately 219–220 mm away; the global seed regressions include closer
placements. The radius threshold is an experimental workspace heuristic.

The rule and acceptance criteria were frozen to disk **before** evaluation on
65000–65049. These placements were not used to choose the threshold. Fifty
placements are reused across four masses and two dynamics modes; do not interpret
all 400 candidate episodes as independent random placements.

## Independent results

Each cell is final successful lifts out of 50. Final success requires two-pad
contact and at least 4 cm lift, as in the unchanged task.

| Mass | Previous, stiffness | Outer wrist, stiffness | Previous, arm effects | Outer wrist, arm effects |
|---|---:|---:|---:|---:|
| 5 g | 50 | 50 | 48 | 49 |
| 10 g | 50 | 50 | 48 | 50 |
| 15 g | 50 | 50 | 48 | 50 |
| 20 g | 50 | 50 | 48 | 50 |

Acceptance requires >=95% final success and <=5% sampled table-contact episodes
in **every mass/dynamics condition**. The candidate meets these observed rates;
this finite sample does not establish a guaranteed population success rate.

The remaining 5 g failure is seed 65004. Both pads contact at steps 84–88, but
contact is lost at 89, before scheduled lift. The gate never releases the arm
to lift. Its full trace and phase images are preserved rather than hidden by
changing success criteria or weakening the gate.

Physics-step audit: 16 candidate episodes, seeds 61010/61014/65000/65001 at 5/20 g
in both dynamics modes. All lift successfully, with **zero robot/table contacts
across 105,600 physics steps**. This smaller audit does not certify every substep
of the full evaluation.

## Validation and artifacts

- Automated tests: **377 passed, one optional LeRobot test skipped**.
- New regression reproduces repaired seed 61010 with arm effects; another verifies
  the adaptive option preserves inner-placement actions. Disabled options still
  reproduce the original scripted controller.
- SHA256 checks confirm source model, calibration, both saved policies, normalizers
  and saved run settings are unchanged.
- `experiments/2026-11-26_pick_lift_adaptive_wrist/`: frozen rule, 800 paired
  comparison episodes, assessment, source manifest/snapshots, physics-step audit,
  pose probe, repaired replay and remaining-failure replay.
- New runner: `scripts/sim2real/analyze/pick_lift_wrist_branch.py`; replay accepts
  `--motion-variant outer_wrist`. No existing default training preset selects it.

## Watch and reproduce

Use a new output directory on every invocation:

```powershell
uv run --locked python -m scripts.sim2real.analyze.replay_pick_lift --controller scripted --mass-kg 0.01 --placement-seed 61010 --calibrated --guarded --arm-effects --motion-variant outer_wrist --viewer --out experiments/outer_wrist_viewer
```

For the remaining failure, use `--mass-kg 0.005 --placement-seed 65004` and another
output directory. Add `--frames` for headless phase PNGs.

Repeat the paired batch with:

```powershell
uv run --locked python -m scripts.sim2real.analyze.pick_lift_wrist_branch --adaptive --out experiments/outer_wrist_repeat
```

The next step is an explicit, serialized training configuration for this candidate
and a short fresh-policy pilot, evaluated separately from these scripted results.
Do not resume an old residual policy and assume it remains compatible: previous
tests show those residuals can introduce table contacts. Arbitrary cube yaw,
gripper calibration, sensing and physical manipulation still need validation.

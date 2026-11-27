# Team review and physical-robot handoff - 2026-11-26

## Current result

The calibrated simulation now completes **reach -> grasp -> lift -> carry ->
place -> release -> retreat**, including a resting-placement check. The validated
script is the reference controller. A new, short SAC residual-learning pilot also
passes the tested full task, but has **not demonstrated a useful advantage** over
the script. Do not launch longer training solely on these results.

Work is on `integration/calibrated-pick-lift`, in draft
[PR #9](https://github.com/drenfr01/rbe501_group_project/pull/9), targeting `main`.
The technical results below are recorded through `ebb31e7`; this handoff adds
documentation only. Main has not been merged or changed by this handoff.

## What changed today

1. Integrated the teammate's measured arm-stiffness calibration and pick/lift
   work, preserving both existing tasks. Fixed Windows import/provenance
   compatibility. Calibration is explicitly selected; it does not silently make
   all existing environments calibrated.
2. Diagnosed acquisition failures rather than immediately retraining. Corrected
   cube-yaw alignment with a limited orientation objective, allowed bounded
   settling before closing/lifting, and used a gentler simulated jaw command.
   Five arm DOF remain; arbitrary full 6-DOF orientation is not imposed.
3. Added guarded carrying and supported placement. Fixed the last lowering
   timeout using bounded Cartesian tracking correction that stops pushing down
   once table support is detected. Release still requires support; failed guards
   do not become successful placements by relaxing tolerances.
4. Added the separate `SO101PickPlace-v1` Gymnasium task and trained a fresh
   **10,000-transition, seed-0 CUDA SAC pilot**. It supplies at most a 1 mm
   Cartesian residual during grasped carry/lowering. Acquisition, gripper,
   release guards and retreat remain scripted. The old reach/pickup networks
   were not joined into this new controller.
5. Compared the frozen full-task pilot against the script under matched small
   carry pushes. Retained both as simulation references; no further training
   followed that comparison.

No actuator-gain retuning was used to repair final placement. Earlier rejected
controllers and failed experiments remain documented for academic provenance;
they are not the recommended final baseline.

## Evidence to review

These are separate evaluation sets, not a pooled independent success estimate.

| Check | Script | SAC | Interpretation |
|---|---:|---:|---|
| Final contact-aware placement regression | 240/240 recorded + 120/120 fresh | Not evaluated in this check | Repairs the final lowering timeout |
| New full-task pilot, paired fresh cases | 60/60 | 60/60 | Pilot works; no success-rate improvement |
| Frozen carry-push evaluation | 320/320 pushed + 40/40 controls | 320/320 pushed + 40/40 controls | Both pass this modest disturbance range |
| Latest automated suite | 457 passed, 1 optional LeRobot skip | Same suite | Local result before this documentation-only handoff |

The paired full-task and push evaluations report zero sampled robot/table
contact episodes and MuJoCo warning episodes. This is sampled simulation
evidence, not a guarantee of collision-free hardware behavior.

Pushes were 0.1/0.2 N along +/-X/Y for 99 ms during early carrying, across ten
new placements and 5/10/15/20 g masses, with nominal cube yaw. Mean pushed final
XY error was **4.829 mm scripted / 4.670 mm SAC**. After accounting for each
controller's matched no-push error, learned-minus-script difference was
**+0.120 mm**, with a 95% placement-cluster interval **[-0.121, +0.365] mm**.
That does not establish improvement. It also does not test recovery from drops.

Review in this order:

- [Environment and viewer instructions](../../docs/pick_place.md).
- [Final placement diagnosis and fix](2026-11-26_place_tracking_fix.md).
- [Full-task environment, reward and pilot results](2026-11-26_pick_place_env_pilot.md).
- [Frozen policy disturbance comparison](2026-11-26_pick_place_disturbance.md).
- [Calibration integration and its limits](2026-11-26_calibration_pick_lift_integration.md).

Raw evaluation evidence is under the corresponding dated `experiments/`
directories. The PR contains substantial recorded evidence; start with these
reports and the control/environment code rather than reviewing trace files first.

## Replaying the current result

From the repository root, with the matching local checkpoint available:

```powershell
uv run --locked python -m scripts.sim2real.analyze.pick_place_pilot --replay --viewer --out experiments/team_policy_review_1
```

Add `--scripted` and choose another new output directory to watch the baseline.
The viewer replays recorded physical poses after the headless rollout. This
command does not train a policy. See the environment documentation for alternate
seeds, masses and yaw offsets.

**Checkpoint sharing is separate from pushing this branch.** The new full-task
checkpoint lives locally under `checkpoints/pick_place_v1_pilot/`; ignored binary
weights, normalizers and replay buffers are not included in the PR. To watch the
same learned controller on another computer, transfer the matching checkpoint
directory, including `seed_0/model.zip`, `seed_0/vecnormalize.pkl`, `run_config.json`
and its associated provenance/configuration files. Preserve the directory layout
and hashes. Include monitor/checkpoint/replay-buffer files when sharing training
history or resuming training, rather than substituting a newly trained policy.
Committed reports/configuration snapshots alone cannot reconstruct the weights.

The existing local CUDA changes in `pyproject.toml` and `uv.lock`, and personal
viewer output directories, are excluded from this handoff commit. Another PC
must independently establish a compatible environment; availability of these
local binaries or GPU packages is not implied by checking out the branch.

## Physical-robot work for the teammate

The next step is a measured hardware comparison, not an assumption that the
simulation transfers unchanged. First review calibration provenance and confirm
joint order, signs, units, zero offsets and the mapping of the measured arm gains
to the real servo settings. The calibration covers five arm joints, with two
stiffness values marked assumed; it does not identify the moving gripper or
contact friction. The successful simulated 0-degree closing command is a model
coordinate, not a validated hardware closing-force instruction.

The current controller uses true simulated cube pose/orientation and contact
state. Hardware needs corresponding sensing/estimation, camera-to-robot/table
transforms, and a way to determine grasp retention and table support before
release. This PR does not provide a verified physical full-task deployment.

Suggested handoff sequence:

1. Review the PR's calibration/control changes and checkpoint provenance.
2. Validate low-speed reach, jaw closing and a short lift separately on the robot.
3. Test the nearby carry, supported release and retreat under supervision, using
   actual robot limits and stopping controls.
4. Record time-aligned commanded/measured joints, cube source/destination poses,
   gripper commands/contact estimates, phase timing, slips/drops and final error.
   Compare matched conditions with simulation before changing gains or training.

Current demonstrated scope is a **25 mm cube**, simulated masses **5-20 g**, the
tested placement/yaw sets and a nearby **12-degree same-radius transfer** (roughly
3-5 cm). It is not arbitrary workspace transport, vision-based perception, or
physical sim2real validation. Only one full-task training seed was evaluated.

If hardware reveals a gap, first determine whether it is calibration, sensing,
grasp/contact modeling or control. Retraining is justified when that gap defines
a measurable learning objective. These results neither prove old RL policies
sufficient for the new task nor prove future learning unnecessary.

## Review status

Keep PR #9 as a draft pending teammate review. No merge, physical operation or
long training run is part of this handoff. This update adds the summary and a
documentation link; it does not change runtime code, models or experiment data.

# Coverage audit: independent cube yaw exposes an acquisition weakness

**A useful next benchmark is now identified and independently replicated:** cube
orientation independent of its position around the robot. Both frozen controllers
pass all 180 sampled workspace-grid lifts, but succeed on only 128/280 initial
and 152/280 independently replicated yaw trials each. Every failed yaw trial
aborts before lift release, so longer training of the current lift-only residual
cannot address these failures through its permitted actions.

No controller, policy, normalization, reward, gains, model, default placement or
training preset changed. These are opt-in initial-pose diagnostics, not a new
training distribution. Base integration commit: `f9004e7`.

## What the existing task assumes

`envs/placement.py::TablePlacer.sample` samples cube radius 0.14–0.22 m and
azimuth +/-40 degrees, then sets cube yaw equal to that azimuth. The aligned
script explicitly assumes this rule: it infers desired jaw yaw from the vector
between the pan axis and cube center. This is documented in the existing
`control/aligned_pick_lift.py` module.

The 33-value pick/lift observation contains measured joints/speeds, command
error, previous action, pinch position, cube center and relative position.
It contains **no cube orientation**. A new regression test verifies that cubes
at the same center with yaw offsets 0 and 30 degrees produce identical initial
observations despite different physical orientations. Later contact can change
observations, so this is a pre-contact observability limitation, not a claim
that orientation could never be inferred from interaction.

The saved policy's residual is masked during acquisition and closing. Even
perfect new policy weights cannot correct a failure that prevents its guard
from releasing lift under this authority structure. Reaching remains a separate
policy; it was not used or altered by these tests.

## Frozen tests and results

The first protocol was written before rollouts, followed by a separate
confirmation protocol repeating the entire yaw ladder on new placements.
Both use calibrated stiffness and the same fixed nominal arm effects, robot
gravity compensation, cube collisions, 0-degree closing endpoint and original
200-step task. Deterministic saved-policy inference uses its matching frozen
observation statistics; no learning updates occur.

| Benchmark | Cases per controller | Scripted successes | SAC successes |
|---|---:|---:|---:|
| Supported workspace grid | 180 | 180 | 180 |
| Initial independent-yaw batch | 280 | 128 | 128 |
| Replicated independent-yaw batch | 280 | 152 | 152 |

The grid covers five radii (0.14, 0.16, 0.18, 0.20, 0.22 m), nine azimuths
(-40 to +40 degrees in 10-degree increments) and four masses (5/10/15/20 g),
with baseline radial cube orientation. Grid reset seed is 83000. Zero sampled
table/initial-approach contacts or warnings. This verifies 45 positions at four
masses, not all possible positions in the continuous workspace.

Yaw batches use normal sampled centers with offsets -45, -30, -15, 0, +15,
+30 and +45 degrees from radial alignment, at all four masses. Initial seeds
82000–82009 and confirmation seeds 84000–84009 are distinct. Each table entry
below contains ten centers repeated across four masses; angles and masses are
correlated repeated conditions, not additional independent placement samples.
For a square cube, -45 and +45 degrees describe equivalent orientations.

| Yaw offset | Initial success, both controllers | Confirmation success, both controllers |
|---:|---:|---:|
| -45 degrees | 4/40 | 8/40 |
| -30 degrees | 6/40 | 8/40 |
| -15 degrees | 34/40 | 40/40 |
| 0 degrees | 40/40 | 40/40 |
| +15 degrees | 36/40 | 40/40 |
| +30 degrees | 4/40 | 8/40 |
| +45 degrees | 4/40 | 8/40 |

All 80 nominal-yaw controls pass for each controller. There are zero paired
final-outcome disagreements between script and SAC in all 1,480 evaluation
episodes. All 152 initial and 128 confirmation failures per controller have
`alignment_timeout`, no closure start and no lift release. The failures occur
in the shared acquisition, rather than an RL lift correction dropping the cube.

The initial yaw batch additionally has **four robot/table contact episodes per
controller**, all failed acquisitions: 10 g, seed 82005 at -45/+30/+45 degrees,
and seed 82006 at +30 degrees. Counts are 85–93 sampled control steps in those
cases. The confirmation batch has no sampled table contacts. Neither batch
has initial approach contacts or MuJoCo warnings. Thus independent yaw is not
ready for promotion to a training task with this controller. These safety
observations are at control boundaries, not every physics substep.

## Contact diagnosis

Three additional scripted replays at seed 82000 use the same sampled mass
12.2764 g and center, with yaw offsets 0/30/45 degrees. The nominal case lifts;
both rotated cases abort with `alignment_timeout`.

At 30/45 degrees, sampled fixed-pad/cube contact starts at step 79, while the
controller is still descending with the jaw open. The trace's phase name
`close` is the nominal clock label, not evidence that closure has started.
The cube moves horizontally up to about 4.17/4.08 mm in these full failed
replays, and the alignment guard never permits closing. The first contact has
roughly 0.3 N normal force per reported contact point; these are solver outputs,
not hardware force measurements. In the wider initial matrix, maximum
pre-close XY cube movement reaches 56.42 mm.

This supports an orientation/contact/acquisition interaction, rather than a
pure reach-IK residual or post-grasp holding weakness. It does not prove that
feeding yaw into the existing solver alone will solve all cases: reachability,
path clearance and gripper seating must still be tested.

## Next experiment, before training

Build a separate orientation-aware scripted acquisition prototype using the
true simulated cube orientation as an explicit **diagnostic oracle**. Test
whether jaw-to-face alignment can improve this frozen benchmark without new
table contacts or degraded nominal-yaw performance. Keep position primary and
constrain only the orientation needed for side grasping; the five-joint arm
should not be forced to satisfy arbitrary six-dimensional poses.

First check feasible grasp poses and open-jaw approach clearance, then run
end-to-end descent/closing/lift. If it works, decide how orientation would be
measured on hardware and represented in observations before training a new
versioned policy. Any acquisition residual would need separately reviewed
authority and contact protections. Do not relax the guard or remove collisions
to turn these failures into apparent successes. No such changes were made here.

## Artifacts and validation

New evaluation module: `scripts/sim2real/analyze/pick_lift_coverage.py`.
New tests: `tests/test_pick_lift_coverage.py`. Tests cover baseline RNG/placement
equivalence, isolated yaw changes, the exact matrix bounds, full 200-step
zero-offset observation/physics equivalence, and initial yaw unobservability.
Tests were introduced before implementation; **423 pass, one optional LeRobot
test skips**. No training or dependency synchronization was performed.

`experiments/2026-11-26_pick_lift_coverage/` contains both protocols, all outcomes,
summaries, source/protected-input provenance, the independent confirmation
driver, yaw-success figure and plotting source, plus three contact traces and
phase images. All protected source-model/calibration/runtime/policy/normalizer
hashes match before and after both sweeps. Original checkpoint binaries remain
local and ignored by Git. User-local CUDA dependency edits remain untouched.

```powershell
uv run --locked python -m scripts.sim2real.analyze.pick_lift_coverage --model-dir checkpoints/calibrated_cartesian_zero_v3_pilot --out experiments/pick_lift_coverage_repeat
```

Use a new output directory. To reproduce the exact confirmation, run its saved
driver from the repository root; its output directory must not already exist:

```powershell
uv run --locked python -c "import runpy; runpy.run_path('experiments/2026-11-26_pick_lift_coverage/confirm_yaw.py', run_name='__main__')"
```

This is a simulated distribution-shift diagnosis with an uncalibrated gripper
and privileged object/contact sensing. It is not hardware grasp validation or
evidence that the original trained policy failed its original yaw distribution.

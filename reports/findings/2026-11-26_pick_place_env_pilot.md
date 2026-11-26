# Full-task Gym environment and fresh CUDA SAC pilot

**The complete-task environment is implemented and validated.** A fresh
10,000-transition CUDA SAC pilot completes **60/60 paired full pick-and-place
trials**, matching the zero-residual scripted baseline at 60/60. No sampled
robot/table contacts or MuJoCo warnings occur. It passes stability; it does
not establish an added practical benefit from RL or authorize long training.

This is a **new residual-learning environment**, not a merge of the old reaching
and pickup networks. The policy supplies small arm corrections during carry and
lowering. The validated script and guards still perform acquisition, opening,
release authorization and retreat. No claim is made that the neural network
learned all those stages from scratch.

## New full-task definition

Registered ID: **`SO101PickPlace-v1`**. Existing `SO101Reach-v0` and
`SO101PickLift-v0` registrations and behavior remain intact. Base integration
commit: `8f36a64`.

- The destination is explicit in the task state and observations. By default,
  use the validated nearby 12-degree same-radius table move. Reset can supply
  `destination_xy`, validated in the existing workspace; that does not certify
  every arbitrary requested path.
- Random training samples retain the 5–20 g mass range, original placement
  band and independent -30/0/+30-degree cube-yaw offsets.
- Maximum full-task horizon: **720 control periods**. Internal pickup uses its
  original 200 periods, then a one-period state machine advances the validated
  carry/lower/release/retreat/settle primitives. The old Gym task is never stepped
  after its truncation. The new outer task has its own termination semantics.
- Actions are three normalized Cartesian residual values, with **<=1 mm vector
  magnitude** per proposal during grasped carry/lower. Vertical residual is
  masked on table support; the policy has no gripper control.
- Residual proposals use the existing IK, joint rate, contact and 4 mm nominal
  table-clearance checks. A rejected pre-execution residual restores the script's
  reference/bias and falls back to one scripted physics period. Actual grasp,
  contact, warning or controller failure terminates the new episode.
- Full success requires completed lift/release/retreat/final observation, a
  supported ungrasped cube within 10 mm XY and 5 mm resting-height error,
  linear speed <=0.01 m/s and angular speed <=0.1 rad/s. Merely lifting, reaching
  the destination while holding, or starting at the destination cannot succeed.
  The original 60-period final observation window remains required.
- Success/failure produces `terminated`; the outer time limit produces
  `truncated`. Stepping after either requires reset.

Flat observations contain **72 values**, including cube quaternion/velocity,
destination/error, original measured arm state, phase/progress/contact/mass,
previous requested residual and reference/bias/axis/milestone state. These require
their own observation-normalization statistics and are incompatible with the old
33-value pickup-policy interface. Cube pose/contact/velocity remain privileged
simulation information; this does not implement hardware perception.

Reward consists of phase-specific distance potential shaping at gamma 0.98,
-0.001 per period, -0.001 times squared applied normalized residual, +10 for
true completed placement and -10 for controller failure. Termination uses zero
absorbing potential; time-limit truncation retains next-state potential for
bootstrapping. Reward components and physical success are reported independently.
Numeric returns should not be compared to the old reaching/pickup reward curves.

## Environment verification

Tests were added first and failed before implementation. **452 tests pass,
one optional LeRobot test skips.** New tests verify:

- Exact per-frame physical qpos equality with the validated standalone sequence
  under zero residual, through the formerly failing 5 g / +30-degree / seed 89000
  complete task, not just acquisition.
- Correct full-success criteria including angular rest, seeded reset and explicit
  goal/orientation observations, truncation/reset requirements and finite actions.
- Masked authority during pickup and bounded authority during transfer.
- Safe restoration and scripted fallback after a rejected residual, without
  advancing physics twice.
- Fresh repository creation, full nested configuration roundtrip, refusal to
  overwrite a run or accept changed configuration.

Both Gymnasium and SB3 environment checkers pass on the actual environment.
`gym.make("SO101PickPlace-v1")` reset/step also passes. CUDA arithmetic is verified
on the RTX 5070 before training; SAC reports device `cuda`. MuJoCo physics remains
CPU simulation.

## Frozen short pilot

Fresh preset `pick_place_residual_v1`, training seed 0, **10,000 transitions**.
SAC retains the existing defaults: learning rate 0.0003, batch 256, gamma 0.98,
two 256-unit hidden layers, replay capacity 300,000, one gradient update per
transition, observation normalization on and reward normalization off. Matching
shaping/training gamma is enforced. No old policy, optimizer or normalizer is
used to initialize it.

The saved model has a 72-value input and three-action output. There are **22
completed training episodes, all successful**, covering 9,918 transitions; the
remaining 82 transitions are the final partial episode, not another completed
success. Checkpoints are saved at 5k and 10k with replay-buffer companions.

The protocol was written before scripted evaluation and training. Scripted
evaluation passed all 60 cases before the pilot was allowed to start. Evaluation
then reloads the new saved model and its own frozen `vecnormalize.pkl`.

| Full-task held-out outcome | Scripted | New SAC residual |
|---|---:|---:|
| Completed placements | 60/60 | **60/60** |
| New paired failures | — | 0 |
| Sampled robot/table contact steps | 0 | 0 |
| MuJoCo warning episodes | 0 | 0 |
| Mean final cube XY error | 4.464 mm | 4.390 mm |
| Maximum final cube XY error | 6.970 mm | 6.744 mm |
| Mean full-task control periods | 457.03 | 460.40 |

Held-out seeds **93000–93004**, four masses 5/10/15/20 g, three yaw offsets
-30/0/+30 degrees: five placement centers repeated over twelve conditions,
not 60 independent positions. Each mass/yaw cell passes 5/5. One training seed
and a small nominal evaluation establish stability only.

The actor actually applies residual corrections on **6,340 held-out periods**,
with zero proposal rejections in this batch. These are complete-task policy
rollouts, not silent substitution of the scripted actions everywhere. Conversely,
the earlier 200-period pickup and protected jaw are still scripted by design.

The small descriptive mean-error change (about 0.073 mm) and slightly longer
episodes do not establish useful RL improvement; no precision-benefit claim or
post-hoc promotion threshold is introduced. Success is already saturated for
the script on this test. The acceptance file passes stability and explicitly
keeps `useful_rl_benefit_established` and `start_long_training` false.

## Saved-policy replay and artifacts

Reloaded learned-policy replay succeeds on a fresh 10 g / nominal yaw / seed
93000 case (4.210 mm XY error), and on the formerly failing 5 g / +30-degree /
seed 89000 case (4.419 mm). Both terminate successfully with release completed,
cube no longer grasped, and zero sampled table contacts/warnings. Their actual
physical trajectories and phase images are saved, not generated by running the
zero-residual baseline instead.

Evidence: `experiments/2026-11-26_pick_place_env_pilot/`, including preregistered
protocol, software/checker/CUDA preflight, paired episode outcomes, training
history, assessment, full run configuration, input/artifact hashes and both
learned-policy replays. Local trained artifacts are in
`checkpoints/pick_place_v1_pilot/`:

- `run_config.json`: task/controller/reward/layout/training/source/model snapshot.
- `seed_0/model.zip`, matching `vecnormalize.pkl` and `train.monitor.csv`.
- Complete checkpoint model/statistics/replay buffer and save log.

Checkpoint binaries remain ignored by Git. Share the whole run directory and
matching code/assets to reproduce this saved policy; pushing the integration
branch does not transfer its weights.

The new repository fingerprints runtime sources, model directory/assets and
calibration. CRLF/LF are normalized for .py/.xml/.json hashes; binary assets are
unchanged. Loading refuses configuration/interface/source drift. Old model,
calibration, policy weights/statistics and protected runtime inputs match before
and after the pilot. The only existing runtime-file edit is registration of the
new environment in `envs/__init__.py`; old controllers/trainers are not modified.
User-local CUDA dependency edits and personal viewer output remain untouched.

New runtime files are `control/pick_place_stepper.py`,
`envs/pick_place_config.py`, `envs/pick_place_env.py`, and
`rl/pick_place_training.py`. The pilot/replay entry point is
`scripts/sim2real/analyze/pick_place_pilot.py`. Usage and task semantics are
documented in `docs/pick_place.md`.

```powershell
# Watch the actual newly trained full-task residual policy
uv run --locked python -m scripts.sim2real.analyze.pick_place_pilot --replay --viewer --model-dir checkpoints/pick_place_v1_pilot --seed 93000 --mass-kg 0.01 --out experiments/full_policy_viewer_1

# Add --scripted to compare its matching zero-residual reference
# Repeat the short pilot only into NEW output/checkpoint directories
uv run --locked python -m scripts.sim2real.analyze.pick_place_pilot --out experiments/full_task_pilot_repeat --model-dir checkpoints/full_task_pilot_repeat
```

The full-task environment and saved-policy workflow are ready for further work.
Choose a measurable learning benefit before a longer multi-seed campaign; this
result does not require more unrelated acquisition diagnosis. No long training,
hardware motion or model/gain adjustment occurred.

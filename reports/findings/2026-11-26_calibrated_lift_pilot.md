# Calibrated residual-lift pilot — 2026-11-26

Steps completed: viewer review, an explicit serialized training preset, and one
fresh 10,000-transition SAC pilot on the RTX 5070. **The pilot fails evaluation;
do not start full training or promote its weights.** The scripted candidate remains
the working baseline.

## Viewer review

Both scripted cases ran in the live viewer with the `outer_wrist` candidate and
arm effects. Seed 61010 at 10 g lifts 5.19 cm. Seed 65004 at 5 g loses stable
contact before lift; the guard never releases. Their new traces are in
`experiments/2026-11-26_pilot_review_repaired/` and
`experiments/2026-11-26_pilot_review_failure/`. Phase PNGs from the earlier wrist
experiment were inspected too. Viewer replay does not change the controller.

## Explicit preset and reconstruction

`src/rbe_501_so101/rl/calibrated_pick_lift.py` adds the separately named
`calibrated_outer_wrist_v1` preset and a run-configuration repository. It records:

- Validated outer-reach wrist rule, gravity command feedforward, face alignment
  under the existing radial cube-yaw placement assumption, and -4 mm waypoint.
- Measured stiffness and fixed nominal arm stiction, relative goal clipping,
  one-step delay and encoder quantization. No new hardware calibration is implied.
- Five-contact-step lift gate, placed beneath residual control so it filters final
  actions. Contact auditing counts robot/table contact at control boundaries.
- Random cube mass 5–20 g, unchanged 25 mm cube, task, reward and residual scale 0.5.
- Resolved environment, scripted and servo/sensing configuration snapshots; model
  and calibration hashes; training settings, seed and transition budget.

The repository refuses an existing run directory rather than automatically
resuming it. Loading verifies the configuration snapshot and model/calibration
checksums. A regression test confirms zero residual reproduces the previously
validated scripted stack for all 200 steps of seed 61010. Legacy pick/lift CLI,
training presets, environment behavior, physical gains, source MJCF and original
checkpoints are untouched.

## Frozen pilot protocol

- One new SAC seed: 0. 10,000 transitions = 50 complete 200-step episodes.
- Existing SAC defaults: learning rate 0.0003, gamma 0.98, batch size 256,
  learning starts 100, buffer size 300,000, one update per step, 256/256 MLP.
- Observation normalization enabled; reward normalization disabled. Model and
  matching normalization state reloaded before deterministic evaluation.
- RTX 5070, PyTorch 2.14.1+cu130 / CUDA 13.0; CUDA arithmetic and SB3 device verified.
  Python 3.12.15, MuJoCo 3.14.0, Gymnasium 1.4.0, SB3 2.9.0, NumPy 2.5.3.
- Fresh held-out placement seeds 66000–66019, four fixed masses, 80 episodes per
  controller. The scripted reference uses the exact training wrapper stack with
  zero residual. Placements are reused across masses, not 80 independent locations.
- Frozen gate: each mass >=95% final success, <=5% table-contact episodes, no
  MuJoCo warnings. No full-training launch follows automatically.

The run configuration, preflight and evaluation protocol were recorded separately
from previous experiments. No old policy, normalizer or replay buffer was reused.

## Results

Cells show successful final lifts / episodes with sampled robot-table contact,
out of 20 episodes at each mass.

| Mass | Scripted reference | Fresh SAC pilot |
|---|---:|---:|
| 5 g | 20 / 0 | 6 / 17 |
| 10 g | 20 / 0 | 7 / 17 |
| 15 g | 20 / 0 | 8 / 15 |
| 20 g | 19 / 0 | 8 / 15 |
| Total / 80 | **79 / 0** | **29 / 64** |

Training history: 47/50 final successes, but 32/50 episodes have sampled table
contacts. These are exploratory rollouts from changing policies, not evidence of
final deterministic performance. No MuJoCo warnings occurred in training or either
evaluation. This pilot establishes that the training pipeline works, not that SAC
has improved the grasp.

The 10 g saved-policy replay at seed 66000 touches the table at step 74, reaches
success at 107, then falls below the success height at 184. The cube is still
grasped and the guard is not aborted; the final lift is only 5.46 mm. Therefore
not every failure is a slip: learned residual motion can undo an achieved lift.
The guard enforces retained contact, not table clearance or a minimum holding
height. The unchanged reward has no explicit table-contact penalty. These are
observed limitations; the small pilot does not establish a unique cause of poor
generalization or show that simply running longer will solve it.

## Files and validation

Automated tests: **382 passed, one optional LeRobot test skipped**. New tests first
failed before the preset implementation, then passed. They cover serialization,
configuration drift, refusing old directories, zero-residual equivalence and
rejecting empty/contact-heavy evaluation evidence. Gymnasium/SB3 environment
checker and reset/step preflight passed. The new saved-policy headless replay
also ran successfully as a diagnostic, with the failed task outcome preserved.

Evidence: `experiments/2026-11-26_calibrated_lift_pilot/` includes protocol,
configuration, versions, all evaluation episodes, 50 training episode rewards and
contact histories, assessment, artifact hashes, and saved-policy replay/PNGs.
Original source model, calibration and both original saved policies/normalizers
passed before/after SHA256 checks.

New pilot artifacts remain **local**, under the existing ignored checkpoints tree:

```text
checkpoints/calibrated_outer_wrist_pilot/
  run_config.json
  seed_0/model.zip
  seed_0/vecnormalize.pkl
  seed_0/train.monitor.csv
  seed_0/checkpoint_log.json
  seed_0/checkpoint/model.zip
  seed_0/checkpoint/vecnormalize.pkl
  seed_0/checkpoint/replay_buffer.pkl
```

The replay buffer is about 90 MB (allocated capacity; only 10,000 transitions
collected). These binary artifacts are not uploaded with the report/code commit;
do not assume a teammate has them. Exact hashes are recorded in `artifacts.json`.

## Commands and next decision

Repeat training only into **new** directories:

```powershell
uv run --locked python -m scripts.sim2real.analyze.calibrated_lift_pilot --out experiments/calibrated_pilot_repeat --model-dir checkpoints/calibrated_pilot_repeat
```

Watch the saved pilot with its own verified preset and frozen normalization:

```powershell
uv run --locked python -m scripts.sim2real.analyze.calibrated_lift_pilot --watch --model-dir checkpoints/calibrated_outer_wrist_pilot --mass-kg 0.01 --placement-seed 66000 --viewer --out experiments/calibrated_pilot_viewer
```

Output directories must be new. Use `--frames` for headless images. The legacy
`run_pick_lift.py` loader does not reconstruct this new preset; use the command above.

Next investigate residual authority during acquisition and holding, clearance
constraints, and the mismatch between exploratory training and deterministic final
evaluation. Freeze any change as another explicit experiment and run a short pilot
on new evaluation placements. Do not replace the scripted baseline, weaken success
criteria, or launch a multi-seed full run based on this result.

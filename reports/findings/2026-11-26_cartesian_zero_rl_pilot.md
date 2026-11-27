# Cartesian acquisition and zero-degree closing: fresh SAC pilot

The new **v3 pilot passes its simulation stability gate**. A freshly trained SAC
policy completes 160/160 held-out lifts, matching the validated scripted controller,
and passes all 24 known regression cases. This is evidence that training can use
the corrected acquisition/closing sequence without the previously observed
regressions. It does not establish a useful improvement from RL or justify a long
training campaign yet.

## Frozen setup

- Base integration commit: `0241310`; separate preset `calibrated_cartesian_zero_v3`.
- Validated Cartesian acquisition, cube-center grasp height, 0-degree moving-jaw
  closing endpoint, measured arm stiffness, robot gravity compensation, enabled
  cube collisions, and fixed nominal arm servo/sensing effects.
- Scripted approach, descent and closing remain protected. RL changes only the
  arm during an established lift; it cannot command the jaw. Residual scale is
  0.05, cumulative command budget 0.02 rad, nominal table clearance 4 mm and
  permitted nominal holding-height drop 0.5 mm. These geometric checks do not
  guarantee safety under every dynamic contact condition.
- The new preset starts the shadow lift reference only after the contact/closing
  guard actually releases. Advancing it at the nominal lift phase while the arm
  was still waiting would create an incorrect reference for residual projection.
  Older controllers and preset implementations remain intact.
- One fresh SAC training seed (0), 10,000 transitions / 50 episodes; all 50
  training episodes ended successfully. Existing SAC defaults are retained:
  learning rate 0.0003, batch 256, learning starts 100, replay capacity 300,000,
  one update per transition, two 256-unit hidden layers, observation normalization
  with reward normalization disabled. Complete configuration is in `run_config.json`.
- Training ran on RTX 5070 with torch 2.14.1+cu130 / CUDA 13.0. Python 3.12.15,
  SB3 2.9.0, Gymnasium 1.4.0, MuJoCo 3.14.0 and NumPy 2.5.3.
  CUDA arithmetic and the SB3 environment checker passed before training.

## Independent evaluation

The protocol was written before training/evaluation. Placement seeds 80000–80019
were paired across controllers, four masses (5, 10, 15, 20 g) and two dynamics
modes (calibrated stiffness alone / stiffness plus nominal arm effects). Thus
160 episodes are **20 placements repeated across eight conditions**, not 160
independent placements. The policy was reloaded with its own frozen observation
statistics. Evaluation uses deterministic policy inference on CPU; training used CUDA.
The task remains the original 200-step horizon with the current 25 mm cube.

| Controller | Stiffness only | With arm effects | Total | Known regressions |
|---|---:|---:|---:|---:|
| Validated scripted reference | 80/80 | 80/80 | 160/160 | See prior closing report |
| Fresh v3 SAC | 80/80 | 80/80 | 160/160 | 24/24 |

Each individual mass/mode condition achieved 20/20. Known seeds 68006, 69005 and
75003 were checked at all four masses in both modes. No controller lost an
already achieved success. Across these rollouts there were zero sampled
robot/table contacts, initial approach/cube contacts, or MuJoCo warnings.
These contact counters are sampled after control steps, not a new physics-substep
audit of this policy. Maximum pre-close XY cube displacement was 0.155 mm for
both paired controllers and 0.141 mm in the known-case policy checks.

The actor was not silently disabled: it applied corrections on 15,333 held-out
control steps, with 89 holding-height rejections, zero clearance rejections and
1,600 budget clips. The closing/acquisition observations match the validated
script in the zero-residual regression tests.

Descriptive motion measurements over the last 30 control steps show mean cube
displacement RMS of 0.2211 mm scripted versus 0.2202 mm learned, while pinch-frame
RMS increases from 0.0010 to 0.0070 mm. RMS is measured relative to the first
sample of that window, **not a target-position error**. All paired cases succeeded,
so no failed cases were dropped. This tiny cube-motion difference from one
training seed is not an established benefit. Nominal success is already saturated.

## Files and reproducibility

New code:

- `src/rbe_501_so101/rl/cartesian_zero_training.py`: serialized preset, guarded
  lift-only residual, fresh-run repository and matching checkpoint loader.
- `scripts/sim2real/analyze/cartesian_zero_pilot.py`: CUDA preflight, fresh pilot,
  paired evaluation, regression checks and replay.
- `tests/test_cartesian_zero_training.py` and `tests/test_cartesian_zero_assessment.py`:
  reconstruction, zero-residual equivalence, delayed-reference and gate checks.

Evidence is under `experiments/2026-11-26_cartesian_zero_rl_pilot/`: protocol,
environment versions, full resolved configuration, episode records and intervention
counts, training history, assessment, summary, source provenance and artifact hashes.
The 5 g / seed 68006 replay reloads the saved policy and ends successfully;
its phase images and trace are under `replay_5g_68006/`.

Local trained artifacts are under
`checkpoints/calibrated_cartesian_zero_v3_pilot/`: run configuration, seed-0
`model.zip`, matching `vecnormalize.pkl`, training monitor, and a complete
checkpoint including replay buffer. Binary checkpoints remain ignored by Git;
sharing this branch alone does not transfer those weights. Share that whole run
directory separately to reproduce the saved policy. The original teammate
policies, previous v1/v2 pilots, model, calibration and validated acquisition
sources passed before/after checksum verification. No existing results were replaced.
User-local CUDA dependency edits remain outside this change.

Validation: `python -B -m pytest -q` reports **414 passed, 1 skipped** (the
optional LeRobot check). `git diff --check` passes. No long training was started.

```powershell
# Repeat the short pilot with NEW output and checkpoint directories
uv run --locked python -m scripts.sim2real.analyze.cartesian_zero_pilot --out experiments/cartesian_zero_pilot_repeat --model-dir checkpoints/cartesian_zero_pilot_repeat

# Watch the locally saved v3 policy; NEW output directory required
uv run --locked python -m scripts.sim2real.analyze.cartesian_zero_pilot --watch --model-dir checkpoints/calibrated_cartesian_zero_v3_pilot --mass-kg 0.005 --placement-seed 68006 --viewer --out experiments/cartesian_zero_v3_viewer
```

Next define a useful objective that the scripted controller does not already
saturate, then run paired robustness/holding tests with frozen success and
contact gates before deciding on longer multi-seed training. Hardware jaw mapping
and contact-force validation remain separate work with the physical robot.

# Pick-and-place with a heavy cube: fixed vs random mass training

The SO-101 picks a 25 mm cube from a random spot on the table and places it on a random goal
(success = cube within 2 cm of the goal, released, and resting). The cube's **mass** is the
experimental parameter. Two residual SAC policies are trained on top of a scripted IK controller:

| Run | Training mass | Script |
|---|---|---|
| **fixed** | always 1.0 kg (the heaviest cube) | `scripts/pick_place_fixed.py` |
| **random** | uniform 0.05–1.0 kg, new each episode | `scripts/pick_place_random.py` |

Both are evaluated on the same starts and goals at 0.05–1.5 kg. A torque analysis explains the results:
the position-controlled servos sag under load (`Δθ = Kp⁻¹ τ`), and the learned correction is too small
to cancel it.

## Folder layout

```
pick_place/
├── README.md                  this file
├── scripts/
│   ├── pick_place_fixed.py    train / watch / compare the fixed-mass run
│   ├── pick_place_random.py   train / watch / compare the random-mass run
│   ├── torque_analysis.py     torque, sag and capacity sweep → 8 figures + report.txt
│   └── record_videos.py       MP4 videos of the results
├── tests/                     pytest tests for everything above
└── report/                    step-by-step results page (open report/index.html)

src/rbe_501_so101/pick_place/  the library code these scripts import
├── env.py                     environment, scripted controller, reward, training, evaluation, CLI
└── torque.py                  equation-of-motion terms, servo sag, payload capacity, feedforward controller
```

Trained models and generated outputs go to `checkpoints/pick-place/`. Only the small results are in git:
the final v2 models (`model.zip`, `vecnormalize.pkl`, training curves), `torque/report.txt`, and the
5-seed comparison `compare_v2_5seeds_50ep.{txt,png}`. Replay buffers and intermediate checkpoints are not.

## What one episode is

One episode = one attempt: pick one cube up and place it on one goal. It lasts 320 control steps of
33 ms (about 10.6 s of simulated time).

At reset the seed draws:

- **cube start**: random point 0.10–0.24 m from the base, ±90° around it;
- **goal**: another random point, at least 6 cm from the start.

The **cube mass** is set per experiment and stays constant during the episode:

| Experiment | Cube mass |
|---|---|
| fixed training | always 1.0 kg |
| random training | new uniform draw from 0.05–1.0 kg each episode |
| evaluation | the mass being tested |

Evaluation episode *i* uses seed 20000 + *i*. Every controller and every mass therefore sees the same
start/goal layouts.

The scripted controller runs these phases (the RL policy adds a small correction, at most ±0.8° per
joint, on top of each step):

| Steps | Phase | What the arm does |
|---|---|---|
| 0–39 | approach | above the cube, jaw open |
| 40–69 | descend | down around the cube |
| 70–99 | close | grip |
| 100–139 | lift | lift the cube |
| 140–219 | carry | to 7 cm above the goal; **steps 200–219 are the hold window** |
| 220–269 | lower | straight down onto the goal |
| 270–289 | release | open the jaw |
| 290–319 | retreat | straight back up |

**Success** (checked at step 320) needs all three:

- the cube was let go within 2 cm of the goal and at most 1.5 cm above the table;
- the cube rests on the table within 2 cm of the goal;
- no finger pad still touches it.

**Placement error** = final horizontal distance from the cube to the goal.

## How the torque numbers are computed

Each number is computed per episode (`_hold_and_load()` in `src/rbe_501_so101/pick_place/torque.py`)
and then averaged over the 50 evaluation episodes at that mass.

| Number | Formula |
|---|---|
| peak torque | `max |τ|` over all 320 steps, τ = MuJoCo `qfrc_actuator` |
| sag | mean of `θ_cmd − θ` over the hold window; the servo holds `τ = Kp (θ_cmd − θ)`, Kp = 20 N·m/rad |
| predicted sag | `Δθ = Kp⁻¹ (g(θ) + m b(θ))`, where `b = 9.81 · J_v,z` at the cube |
| cube drop | mean of (planned carry height − cube height) over the hold window = arm sag + slip in the jaws |
| success % | share of the 50 episodes that succeed |

Example at 1 kg: the shoulder sags 4.41° = 0.077 rad, so it holds 20 × 0.077 ≈ 1.54 N·m. The cube
hangs 22.5 mm low: 21.3 mm from the arm and 1.2 mm of slip.

## Setup

Run every command from the **project root** (the folder above `pick_place/`).

```bash
uv sync                      # installs Python deps, including the dev group (pytest, matplotlib)
```

If `uv` is not on your `PATH`, use its full path (e.g. `~/.local/bin/uv`). Recording videos also needs
`ffmpeg` and a GPU driver with EGL (headless MuJoCo rendering).

## 1. Watch the task

```bash
# scripted controller, 0.5 kg cube, in the MuJoCo viewer
uv run python pick_place/scripts/pick_place_fixed.py watch --scripted --mass-kg 0.5

# a trained policy (seed 0) on a chosen start and goal (radius m, angle deg)
uv run python pick_place/scripts/pick_place_fixed.py watch --mass-kg 1.0 --start 0.18 30 --goal 0.15 -45

# no viewer, just print success and error
uv run python pick_place/scripts/pick_place_random.py watch --mass-kg 1.0 --episodes 10 --headless
```

## 2. Train

```bash
uv run python pick_place/scripts/pick_place_fixed.py  train --seeds 0 1 2 3 4
uv run python pick_place/scripts/pick_place_random.py train --seeds 0 1 2 3 4
```

- 300k steps per seed by default (`--timesteps` sets the total). Rerunning resumes from the last
  checkpoint, saved every 25k steps and on Ctrl-C.
- Runs are saved in `checkpoints/pick-place/pick_place_fixed_1000g_25mm_v2/` and
  `pick_place_random_50_1000g_25mm_v2/`, one `seed_N/` folder each.
- Add `--cpu` to train on the CPU even if a GPU is present.

## 3. Compare scripted vs fixed vs random

```bash
uv run python pick_place/scripts/pick_place_fixed.py compare --episodes 50 --plot checkpoints/pick-place/compare.png
```

Runs the scripted controller and every trained v2 run on the same seeded starts and goals at each
mass (default 0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5 kg) and prints success rate and placement error.

## 4. Torque analysis

```bash
OMP_NUM_THREADS=1 uv run python pick_place/scripts/torque_analysis.py             # full sweep, ~4 min on 20 cores
uv run python pick_place/scripts/torque_analysis.py --episodes 5                  # quick look
uv run python pick_place/scripts/torque_analysis.py --plot-only                   # redraw from the saved sweep
```

Five controllers × seven masses × 50 episodes. Writes `checkpoints/pick-place/torque/`:
`report.txt` (all tables), `sweep.pkl` (raw results), and `fig1`–`fig8`:

| Figure | Shows |
|---|---|
| fig1_torque_breakdown | actuator torque split into gravity, contact, inertial, Coriolis terms over one episode |
| fig2_model_validation | payload torque and sag predicted by the model vs measured |
| fig3_sag_and_drop | joint sag and cube drop vs mass |
| fig4_capacity | largest payload over the workspace (stall and rated torque) |
| fig5_load_vs_mass | peak torque, saturation, heat and work vs mass |
| fig6_correction | how much each controller corrects vs the sag it should cancel |
| fig7_success | success rate and placement error vs mass |
| fig8_residual_by_phase | where in the episode the RL policies act |

`OMP_NUM_THREADS=1` stops each worker process from starting its own thread pool.

## 5. Videos

```bash
uv run python pick_place/scripts/record_videos.py                      # all videos, ~1 min
uv run python pick_place/scripts/record_videos.py --only torque_rl_1kg_seed4
```

Writes MP4s to `checkpoints/pick-place/videos/`. Green disk = 2 cm place tolerance; red sphere =
where the cube would be without sag.

| Video | Shows |
|---|---|
| compare_1kg | scripted, feedforward, fixed SAC, random SAC side by side at 1.0 kg |
| compare_1500g | the same at 1.5 kg (beyond the training range) |
| limits_1500g | an episode where both SAC policies miss and feedforward still places it |
| mass_sweep | scripted controller at 0.05 / 0.5 / 1.0 / 1.5 kg |
| torque_1kg | scripted vs feedforward with live joint torque and sag charts |
| torque_rl_1kg | fixed vs random SAC with live charts (training seed 0) |
| torque_rl_1kg_seed4 | the same episode with training seed 4: fixed places, random misses |

## 6. Results page

Open `pick_place/report/index.html` in a browser. It walks through the results in 12 steps
(arrow keys or Next/Back); `details.html` has every table and the code walkthrough. The figures and
videos it shows are copies of the ones in `checkpoints/pick-place/`; after re-running steps 4–5, copy
them over again:

```bash
cp checkpoints/pick-place/torque/fig*.png checkpoints/pick-place/videos/*.mp4 pick_place/report/
```

The numbers in the page's charts are typed into `report/index.html` (the `D = {...}` block) from
`report.txt`.

## 7. Tests

```bash
uv run pytest pick_place/tests -q          # this folder only
uv run pytest -q                           # whole project
```

## Main results (50 episodes per mass)

| Mass (kg) | Scripted | Fixed SAC | Random SAC | Feedforward (knows the mass) |
|---|---|---|---|---|
| 0.5 | 98% | 100% | 94% | 100% |
| 0.75 | 84% | 94% | 84% | 100% |
| 1.0 | 60% | 86% | 80% | 98% |
| 1.25 | 42% | 70% | 58% | 96% |
| 1.5 | 44% | 62% | 52% | 92% |

- At 1 kg the shoulder sags 4.4° and the cube hangs 22.5 mm below plan; the shoulder uses 68% of its
  stall torque.
- The spring model `Δθ = Kp⁻¹ (g(θ) + m b(θ))` predicts the simulated sag with R² ≈ 1.
- The residual policy can shift the command by at most 0.2 × 0.07 rad = 0.8°, far less than the 4–5°
  of sag, so both policies only nudge. Fixed matches or beats random at every mass
  (pooled 0.75–1.5 kg: 78% vs 69%, p ≈ 0.03).
- Feedforward adds the predicted sag to the command. It is an upper bound, not a fair competitor:
  it is given the true mass.

# First retrain on the calibrated sim: much better, but it learned to dither (2026-11-01)

**In one sentence:** the policy trained on `residual_ik_sim2real` (commit `8ec9145`, 3 seeds × 1M steps) cuts the
deployment error from 8.9 to 4.0 mm and beats IK hold and gravity compensation everywhere. It is rejected on 4 of 5
acceptance criteria, though, and it gets its accuracy by dithering every command at the maximum rate. That would
buzz and wear the real servos, and it relies on the friction model in a regime the gate never tested.

## Result (calibrated sim, mean (p95) holding error in mm, 20 episodes per condition)

| condition | retrained | previous policy | IK hold | gravity comp | integral |
|---|---|---|---|---|---|
| deployment | 3.96 (8.94), 54% saturated | 8.92 | 20.41 | 12.42 | 0.80 |
| kp 0.7× | 5.89 (11.50), 48% | 15.01 | 28.14 | 14.22 | 0.81 |
| kp 1.4× | 7.49 (17.39), 49% | 7.03 | 15.97 | 12.99 | 0.81 |
| friction band 1.5× | 4.35 (8.47), 52% | 8.37 | 23.34 | 18.34 | 0.81 |

The verdict is **REJECT**:

- the deployment error is 3.96 mm against a 2 mm limit;
- 54% of holding steps are saturated, against a 20% limit;
- the worst stress case is 7.49 mm (kp 1.4×), against a 5 mm limit;
- it does worse than integral control.

It does beat IK hold by 16.5 mm, well beyond the 1.1 mm spread between seeds. Seed means are 3.5–4.6 mm. Data:
`experiments/2026-11-01_acceptance_reach_1m_residual_ik_sim2real/`.

## The policy dithers

Over the holding window (deployment condition, seeds 0 and 2):

- **Commands:** every joint's command moves 2.0° per step on average, which is the full hardware rate limit of 0.035
  rad. The action reverses sign on 84–94% of steps for pan and shoulder lift.
- **Measured motion:** shoulder lift and elbow move only 0.05–0.12° per step, because friction absorbs the dither.
  Wrist flex and roll move about 0.7–0.8° per step, so the wrist vibrates at the control rate.

Dithering is a known way to beat static friction, so the policy found a real trick in the model. It doesn't
transfer safely, though, for three reasons:

1. On hardware this is a 2° goal reversal at up to 15 Hz on every servo: audible buzzing, heating, and gear wear.
2. The gate validated the friction model only for slow, quasi-static approaches. How the real servo responds to
   fast reversals (its internal control loop, the deadband, the motor's bandwidth) was never measured, so the sim's
   prediction for a dithering policy is untrustworthy.
3. Half of the saturation is on wrist flex and wrist roll, which barely affect the gripper position. That is
   chatter with no purpose.

The reward's `smoothness_penalty` (0.01 per unit of action change) is too weak to discourage this.

## Fix: a much stronger smoothness penalty

I swept the smoothness weight on single 200k-step runs (seed 0, deployment condition, 20 episodes):

| weight | error (mm) | p95 (mm) | saturated | command change (°/step) | reversals |
|---|---|---|---|---|---|
| 0.01 (old) | 6.52 | 11.07 | 18% | 1.91 | 98% |
| 0.1 | 6.65 | 14.25 | 35% | 1.89 | 100% |
| 0.3 | 11.51 | 18.29 | 26% | 1.66 | 97% |
| 1 | 5.54 | 9.86 | 20% | 1.06 | 95% |
| **3** | **3.50** | 8.31 | 15% (wrist roll only) | **0.03** | 60% |
| 10 | 3.99 | 7.30 | 45% (elbow at the residual cap) | 0.01 | 12% |
| 30 | 6.62 | 20.64 | 30% | 0.00 | 7% |
| 100 | 16.80 | 32.43 | 65% | 0.00 | 4% |

Weights below 1 can't work because the action is scaled: one unit of action is 0.2 rad of residual, so a 1°
dither is an action change of only about 0.09, which costs about 0.04 per step at weight 1. That is tiny next to
the distance terms. At weight 3 the dither disappears: the remaining reversals are below one encoder count. With
only 200k steps, weight 3 is already more accurate (3.5 mm) than the 1M-step dithering run. Holding a steady
correction costs nothing under this penalty, so it suppresses chatter without discouraging sag correction. Above
weight 10 the policy gives up accuracy. Single-seed short runs are noisy (the 0.3 row), so treat differences of a
few millimetres as unresolved.

`residual_ik_sim2real_smooth` is `residual_ik_sim2real` with a smoothness weight of 3.

`AcceptanceConfig` gains a sixth criterion, added before the second retrain: the mean command change per holding
step must be at most 0.25° (about three encoder counts). The dithering policy moves 1.92°/step. Integral control
moves 0.01°/step, and IK hold and gravity compensation move 0.00.

## Second full retrain: `residual_ik_sim2real_smooth` (commit `c8baf43`, 3 seeds × 1M steps)

Calibrated sim, mean (p95) holding error in mm, 20 episodes per condition
(`experiments/2026-11-01_acceptance_reach_1m_residual_ik_sim2real_smooth/`):

| condition | smooth retrain | first retrain | IK hold | gravity comp | integral |
|---|---|---|---|---|---|
| deployment | **1.76** (4.56), 0.05°/step | 3.96, 1.92°/step | 20.41 | 12.42 | 0.80 |
| kp 0.7× | 2.74 (7.66) | 5.89 | 28.14 | 14.22 | 0.81 |
| kp 1.4× | 4.08 (9.48) | 7.49 | 15.97 | 12.99 | 0.81 |
| friction band 1.5× | 2.28 (5.30) | 4.35 | 23.34 | 18.34 | 0.81 |

The verdict is **REJECT on 2 of 6 criteria**. It passes four:

- deployment error 1.76 mm, against a 2 mm limit;
- beats IK hold by 18.7 mm, against a seed spread of 0.34 mm;
- no dithering, at 0.05°/step;
- worst stress case 4.08 mm, against a 5 mm limit.

It fails two:

- **Saturation, 23.5% against a 20% limit.** All of it is wrist roll, which is pinned at its ±0.2 rad residual limit
  in seeds 0 and 2 (25% and 46% of holding steps). Seed 1 has none, and every other joint is at 0%. A pinned roll
  correction moves the gripper by at most about 1.6 mm (the gripper site sits 8 mm off the roll axis), and it is
  constant, not chatter. So this failure reflects how the criterion counts saturation (any joint), not a control
  problem. The criterion is left unchanged, because it was fixed before the run.
- **No worse than integral control: it is worse in every condition** (1.76 against 0.80 mm at deployment).

What this shows:

1. The smoothness penalty fixed the dithering without costing accuracy. Error fell from 3.96 to 1.76 mm, so the
   accuracy of the first retrain didn't actually depend on dithering.
2. The policy is the first learned controller to meet the 2 mm deployment limit in the calibrated sim, and it stays
   within 5 mm across the stiffness and friction range. That is a feedback-like robustness the original policy
   (D-08) did not have.
3. Integral feedback is still about 2× more accurate in sim, so the paper's comparison is between a learned
   corrector and a simple integrator. Which is better on the real arm (settling, behaviour under real friction) is
   the open question for Phase 3.

# Training assumptions of the reach policy and how each was tested

Policy: `checkpoints/reach_1m_residual_ik` (SAC, 1M steps, seeds 0 / 1 / 2, preset `residual_ik`, trained on nominal
dynamics without randomization). Nominal sim result: 0.3 mm mean, 0.7 mm p95.
Tests run in sim by `scripts/sim2real/analyze/robustness_sweep.py`
([results](../experiments/2026-10-25_reach_robustness/README.md)), 20 episodes per seed and condition.

| ID | Assumption | How tested | Result | Holds? |
|---|---|---|---|---|
| A-01 | Actuator stiffness equals the MJCF placeholder | kp scaled 0.3x to 3x on every actuator | 0.34 mm nominal; under 2 mm only for 0.75x to 1.5x; 22 mm at 0.3x | only within about -25% / +50% (D-01, D-08) |
| A-02 | Joint angles are measured exactly | 4096-count encoder rounding | 0.34 to 1.00 mm | yes, with a 3x cost |
| A-03 | No delay between command and measurement | rounding + one 33 ms step delay | 14.35 mm, worse than IK hold (10.37 mm); 39% saturated | **no** (D-05) |
| A-04 | Arm starts at home, at rest | real backend driving the sim arm (ramp + settle) | baselines unchanged; policy pending Phase 3 | pending |
| A-05 | Commands move at the training rate limit | half speed with rounding + delay | 14.35 to 10.61 mm | partly |
| A-06 | Any target in the sim joint range is reachable | 400 sampler targets vs real calibrated range | 4.8% need elbow or shoulder lift outside the hardware range under the Phase 1 mapping (9.5% assuming identity) | **no** (D-02, D-11) |
| A-07 | Joints have no friction | not testable in sim: MuJoCo friction creeps to zero hysteresis | - | pending Phase 0 (D-03) |

## Transfer decision (provisional)

**Retrain.** On the nominal sim with encoder rounding and a one-step delay the policy fails all three checks:
mean error 14.35 mm against a 2 mm limit, worse than IK hold, and 38.6% action saturation against 5%. Even without
delay it saturates on 18% of holding steps at nominal stiffness. The verdict is provisional until the Phase 0
calibration exists; rerun the sweep with `--calibration` and `--session` then.

The comparison controllers frame the result. Integral control on joint angles holds 0.77 to 0.85 mm under every
condition tested (stiffness 0.3x to 3x, rounding, delay, half speed). It uses no model, so it is the reference the
policy has to beat on hardware. Gravity compensation matches it only at nominal stiffness.

## What a retrain changes

`PRESETS["residual_ik_calibrated"]` trains around the measured stiffness with kp randomization, encoder rounding,
and a random 0 to 1 step delay. It needs `data/calibration/actuator_calibration.json` from Phase 0. Targets outside
the hardware joint range (A-06) should also be excluded from training.

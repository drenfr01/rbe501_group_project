# Acceptance in the calibrated sim (`residual_ik_sim2real` conditions)

Checkpoint `checkpoints/reach_1m_residual_ik_sim2real`, seeds [0, 1, 2].

## Verdict: REJECT

- FAIL: deployment mean error 3.96 mm vs limit 2.00 mm
- PASS: beats IK hold by 16.45 mm vs seed spread 1.09 mm
- FAIL: saturation 54.2% vs limit 20%
- FAIL: dither: command moves 1.92 deg/step while holding vs limit 0.25
- FAIL: worst case kp_scale 1.4: 7.49 mm vs limit 5.00 mm
- FAIL: vs integral control, worst at kp_scale 1.4: +6.68 mm (allowed +0.00 mm)

Holding-window error, mean (p95) in mm, over the same evaluation targets; °/step = mean command change per holding step; sat = policy saturation.

| condition | trained | ik_hold | gravity_comp | integral |
|---|---|---|---|---|
| deployment 1 | 3.96 (8.94, 1.92°/step, sat 54%) | 20.41 (31.50, 0.00°/step) | 12.42 (20.52, 0.00°/step) | 0.80 (1.02, 0.01°/step) |
| kp_scale 0.7 | 5.89 (11.50, 1.89°/step, sat 48%) | 28.14 (41.66, 0.00°/step) | 14.22 (25.67, 0.00°/step) | 0.81 (1.15, 0.01°/step) |
| kp_scale 1.4 | 7.49 (17.39, 1.91°/step, sat 49%) | 15.97 (27.33, 0.00°/step) | 12.99 (27.36, 0.00°/step) | 0.81 (1.03, 0.01°/step) |
| stiction_scale 1.5 | 4.35 (8.47, 1.92°/step, sat 52%) | 23.34 (40.73, 0.00°/step) | 18.34 (30.14, 0.00°/step) | 0.81 (1.09, 0.01°/step) |

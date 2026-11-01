# Acceptance in the calibrated sim (`residual_ik_sim2real` conditions)

Checkpoint `checkpoints/reach_1m_residual_ik_sim2real_smooth`, seeds [0, 1, 2].

## Verdict: REJECT

- PASS: deployment mean error 1.76 mm vs limit 2.00 mm
- PASS: beats IK hold by 18.65 mm vs seed spread 0.34 mm
- FAIL: saturation 23.5% vs limit 20%
- PASS: dither: command moves 0.05 deg/step while holding vs limit 0.25
- PASS: worst case kp_scale 1.4: 4.08 mm vs limit 5.00 mm
- FAIL: vs integral control, worst at kp_scale 1.4: +3.27 mm (allowed +0.00 mm)

Holding-window error, mean (p95) in mm, over the same evaluation targets; °/step = mean command change per holding step; sat = policy saturation.

| condition | trained | ik_hold | gravity_comp | integral |
|---|---|---|---|---|
| deployment 1 | 1.76 (4.56, 0.05°/step, sat 24%) | 20.41 (31.50, 0.00°/step) | 12.42 (20.52, 0.00°/step) | 0.80 (1.02, 0.01°/step) |
| kp_scale 0.7 | 2.74 (7.66, 0.04°/step, sat 23%) | 28.14 (41.66, 0.00°/step) | 14.22 (25.67, 0.00°/step) | 0.81 (1.15, 0.01°/step) |
| kp_scale 1.4 | 4.08 (9.48, 0.09°/step, sat 24%) | 15.97 (27.33, 0.00°/step) | 12.99 (27.36, 0.00°/step) | 0.81 (1.03, 0.01°/step) |
| stiction_scale 1.5 | 2.28 (5.30, 0.05°/step, sat 23%) | 23.34 (40.73, 0.00°/step) | 18.34 (30.14, 0.00°/step) | 0.81 (1.09, 0.01°/step) |

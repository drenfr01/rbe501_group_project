# Acceptance in the calibrated sim (`residual_ik_sim2real` conditions)

Checkpoint `checkpoints/smoke_residual_ik_sim2real`, seeds [0].

## Verdict: REJECT

- FAIL: deployment mean error 10.81 mm vs limit 2.00 mm
- PASS: beats IK hold by 9.59 mm vs seed spread 0.00 mm
- PASS: saturation 17.5% vs limit 20%
- FAIL: worst case kp_scale 0.7: 14.25 mm vs limit 5.00 mm
- FAIL: vs integral control, worst at kp_scale 0.7: +13.44 mm (allowed +0.00 mm)

Holding-window error, mean (p95) in mm, over the same evaluation targets; sat = policy saturation.

| condition | trained | ik_hold | gravity_comp | integral |
|---|---|---|---|---|
| deployment 1 | 10.81 (37.39, sat 18%) | 20.41 (31.50) | 12.42 (20.52) | 0.80 (1.02) |
| kp_scale 0.7 | 14.25 (33.21, sat 18%) | 28.14 (41.66) | 14.22 (25.67) | 0.81 (1.15) |
| kp_scale 1.4 | 12.67 (41.56, sat 20%) | 15.97 (27.33) | 12.99 (27.36) | 0.81 (1.03) |
| stiction_scale 1.5 | 10.71 (40.43, sat 18%) | 23.34 (40.73) | 18.34 (30.14) | 0.81 (1.09) |

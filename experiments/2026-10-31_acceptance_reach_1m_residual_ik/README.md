# Acceptance in the calibrated sim (`residual_ik_sim2real` conditions)

Checkpoint `checkpoints/reach_1m_residual_ik`, seeds [0, 1, 2].

## Verdict: REJECT

- FAIL: deployment mean error 8.92 mm vs limit 2.00 mm
- PASS: beats IK hold by 11.48 mm vs seed spread 2.24 mm
- FAIL: saturation 30.9% vs limit 20%
- FAIL: worst case kp_scale 0.7: 15.01 mm vs limit 5.00 mm
- FAIL: vs integral control, worst at kp_scale 0.7: +13.62 mm (allowed +0.00 mm)

Holding-window error, mean (p95) in mm, over the same evaluation targets; sat = policy saturation.

| condition | trained | ik_hold | gravity_comp | integral |
|---|---|---|---|---|
| deployment 1 | 8.92 (18.15, sat 31%) | 20.41 (31.50) | 12.42 (20.52) | 0.80 (1.02) |
| kp_scale 0.7 | 15.01 (26.72, sat 40%) | 28.14 (41.66) | 14.22 (25.67) | 1.39 (4.35) |
| kp_scale 1.4 | 7.03 (14.85, sat 32%) | 15.97 (27.33) | 12.99 (27.36) | 0.81 (1.03) |
| stiction_scale 1.5 | 8.37 (17.89, sat 31%) | 23.34 (40.73) | 18.34 (30.14) | 0.81 (1.09) |

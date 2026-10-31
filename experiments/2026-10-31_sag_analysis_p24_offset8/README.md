# Phase 0 sag check: rbe501_follower

5 poses, each approached from above and from below. Sag is commanded minus
measured (LeRobot degrees); hysteresis is half of (from above minus from below).

| joint | mean sag (deg) | max abs sag (deg) | mean abs hysteresis (deg) | MJCF kp | fitted kp (95% CI) | friction (N*m) |
|---|---|---|---|---|---|---|
| shoulder_pan | 0.08 | 0.17 | 0.99 | 20.0 | not identifiable (torque span 0.000 N*m) | - |
| shoulder_lift | -1.76 | 2.69 | 1.53 | 20.0 | 10.22 (8.95 to 11.92) | 0.273 |
| elbow_flex | -2.08 | 2.19 | 1.80 | 20.0 | 11.14 (10.45 to 11.94) | 0.350 |
| wrist_flex | -0.49 | 0.61 | 1.07 | 6.0 | 11.93 (9.07 to 17.40) | 0.223 |
| wrist_roll | 0.05 | 0.17 | 0.86 | 4.0 | not identifiable (torque span 0.000 N*m) | - |

**Early warning** (sag above 50% of the policy's ±0.1 rad correction cap, i.e. 2.86 deg):
no joint exceeds it; the existing policy should have correction headroom.

Source session: `/workspaces/rbe501_group_project/data/sag_check/2026-10-31_rbe501_follower_p24_192311`. Calibration written to `/workspaces/rbe501_group_project/experiments/2026-10-31_sag_analysis_p24_offset8/actuator_calibration.json`.

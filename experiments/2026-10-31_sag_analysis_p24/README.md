# Phase 0 sag check: rbe501_follower

12 poses, each approached from above and from below. Sag is commanded minus
measured (LeRobot degrees); hysteresis is half of (from above minus from below).

| joint | mean sag (deg) | max abs sag (deg) | mean abs hysteresis (deg) | MJCF kp | fitted kp (95% CI) | friction (N*m) |
|---|---|---|---|---|---|---|
| shoulder_pan | -0.03 | 0.18 | 0.74 | 20.0 | not identifiable (torque span 0.000 N*m) | - |
| shoulder_lift | -1.98 | 2.77 | 0.63 | 20.0 | 15.79 (13.07 to 19.95) | 0.174 |
| elbow_flex | -1.89 | 2.77 | 0.49 | 20.0 | 12.27 (10.61 to 14.54) | 0.106 |
| wrist_flex | -0.34 | 0.58 | 0.81 | 6.0 | not identifiable (torque span 0.050 N*m) | - |
| wrist_roll | 0.01 | 0.05 | 0.83 | 4.0 | not identifiable (torque span 0.000 N*m) | - |

**Early warning** (sag above 50% of the policy's ±0.1 rad correction cap, i.e. 2.86 deg):
no joint exceeds it; the existing policy should have correction headroom.

Source session: `/workspaces/rbe501_group_project/data/sag_check/2026-10-31_rbe501_follower_p24_190722`. Calibration written to `/workspaces/rbe501_group_project/data/calibration/actuator_calibration.json`.

# Phase 2: sim robustness of the reach policy

Checkpoint `checkpoints/reach_1m_residual_ik`, seeds [0, 1, 2]. Errors are holding-window means (mm),
p95 in parentheses; baselines are deterministic and evaluated once.

**Provisional:** no Phase 0 calibration was given, so the decision uses the placeholder MJCF.

## Verdict: RETRAIN

Condition: nominal (placeholder kp) sim, encoder rounding, 1-step delay.

- FAIL: mean error 14.35 mm vs limit 2.00 mm
- FAIL: beats IK hold by -3.97 mm vs seed spread 3.70 mm
- FAIL: saturation 38.6% vs limit 5%

## Feedback or feedforward: feedforward

Growth = worst error over the sweep / error at nominal kp. Policy 65.06x, integral control 1.00x, gravity compensation 32.49x (position 1.20 on a log scale: 0 = like integral, 1 = like gravity comp).

## Working stiffness range

The policy meets the transfer error limit for kp scales 0.75 to 1.5. No Phase 0 stiffness measured yet.

## Stiffness sweep (every actuator kp scaled)

| kp_scale | trained | ik_hold | gravity_comp | integral | policy saturation |
|---|---|---|---|---|---|
| 0.3 | 22.42 (37.25) | 34.84 (58.79) | 24.95 (41.39) | 0.76 (0.90) | 35.7% |
| 0.5 | 6.29 (10.67) | 20.86 (33.27) | 10.54 (16.71) | 0.76 (0.93) | 10.0% |
| 0.75 | 1.59 (2.72) | 13.77 (21.53) | 3.59 (5.46) | 0.77 (0.93) | 11.7% |
| 1 | 0.34 (0.66) | 10.29 (15.92) | 0.77 (0.93) | 0.77 (0.93) | 18.3% |
| 1.5 | 1.77 (3.16) | 6.86 (10.48) | 3.40 (5.28) | 0.77 (0.93) | 30.0% |
| 2 | 3.28 (6.51) | 5.18 (7.82) | 5.04 (7.81) | 0.77 (0.93) | 31.7% |
| 3 | 6.51 (14.51) | 3.51 (5.22) | 6.69 (10.31) | 0.77 (0.93) | 26.1% |

## Sensing and timing (encoder rounding, delay in control steps)

| delay_steps | trained | ik_hold | gravity_comp | integral | policy saturation |
|---|---|---|---|---|---|
| 0 | 1.00 (2.69) | 10.37 (16.27) | 0.81 (0.92) | 0.85 (1.12) | 18.6% |
| 1 | 14.35 (33.66) | 10.37 (16.27) | 0.81 (0.92) | 0.85 (1.12) | 38.6% |

## Reduced command speed

| speed_scale | trained | ik_hold | gravity_comp | integral | policy saturation |
|---|---|---|---|---|---|
| 0.5 | 10.61 (23.73) | 10.37 (16.27) | 0.81 (0.92) | 0.85 (1.12) | 33.8% |
| 1 | 14.35 (33.66) | 10.37 (16.27) | 0.81 (0.92) | 0.85 (1.12) | 38.6% |

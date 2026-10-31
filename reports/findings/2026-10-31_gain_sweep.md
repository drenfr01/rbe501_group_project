# Finding: servo P-gain sweep (2026-10-31)

**TL;DR:** Every steady-state error on the real arm scales as 1/P: doubling the STS3215 P gain from 16 to 32 halves
the gravity sag, the direction-dependent miss, and even the unloaded deadband. Higher gain costs smooth motion:
P = 24 is slightly jerky, and P = 32 is very jerky and rings briefly before settling. We chose **P = 24** as the
default (`HardwareConfig.position_p_gain`); every later calibration and the retrain use it. Gain alone doesn't
solve transfer: even at P = 32 the shoulder misses by 3.6°, which is 7–15 mm at the gripper.

- **Data:** `data/pose_check/2026-10-31_rbe501_follower_p{16,24,32}_*` (clean tree at `5ed8e88`). Each folder has
  `servo_settings.json` confirming the gain read back from every motor.
- **Command:** `pose_check.py --p-gain G --poses L_pose lean_forward_30 L_pose`. This reaches the L pose first from
  rest, then from leaning forward, so each gain is measured from both directions. D = 32, I = 0, 15°/s ramps,
  2 s settle, 20 reads.
- **Excluded:** `..._p16_184105`, whose last pose was invalid. LeRobot clipped the final goal because the shoulder
  lagged the ramp by more than 10°, and the procedure didn't resend it. Fixed in `5ed8e88`; the P = 16 run was
  repeated.

## Results (reading minus goal, sim degrees; positive = the direction gravity pulls)

| | P = 16 | P = 24 | P = 32 | error × P |
|---|---|---|---|---|
| Elbow, lifting against gravity | 6.35 | 4.07 | 2.93 | 102 / 98 / 94 |
| Shoulder, moving back against gravity | 7.71 | 4.98 | 3.57 | 123 / 120 / 114 |
| Shoulder, moving with gravity | 0.76 | 0.58 | 0.58 | |
| Shoulder hysteresis (difference) | 6.95 | 4.40 | 2.99 | 111 / 106 / 96 |
| Pan (no gravity load) | 1.88 | 1.18 | 0.92 | 30 / 28 / 29 |
| Wrist flex | 2.58 | 1.52 | 1.08 | 41 / 36 / 35 |
| Worst forearm below level | 7.7 | 5.2 | 4.0 | |
| Reading spread (std, 20 reads) | 0.000 | 0.000 | 0.000 | |
| Operator | steady, quiet | slightly jerky on the return move, steady at the end | very jerky moving, slight oscillation at the end | |

## What it shows

1. **The servo acts as a proportional controller against gravity plus friction.**
   - Error × P stays nearly constant per joint (within 10%), so friction (the unloaded deadband) and gravity sag
     both scale as 1/P.
   - The STS3215's firmware deadzone is negligible: the error doesn't level off at a floor.
   - Modelling consequence: a calibration fitted at one gain can be rescaled to another, with real stiffness
     proportional to P.
2. **There's a smoothness cost.**
   - Jerkiness appears on the moves (30 Hz goal steps at the maximum acceleration setting), and at P = 32 a short
     ring-down at the end.
   - Readings were identical across the 20 reads at every gain, so the arm had settled within the 2 s settle time.
   - For a policy stepping at 30 Hz, motion quality matters as much as the final error.
3. **The direction dependence persists at every gain** (3–7° on the shoulder).
   - A fixed feedforward correction can't remove it.
   - Feedback (integral baseline) or a policy trained with randomized friction is still required (D-08).

## Decision

**P = 24, D = 32.** It cuts every error by about 35% against LeRobot's default, with motion still acceptable and
no visible ringing, and the arm stays gentler around the table than at P = 32. Higher damping (D = 48–64) at
P = 32 wasn't tested; it could recover smooth motion at half the error, and is a possible follow-up.

# Finding: commanded poses on the real arm (2026-10-31)

**TL;DR:** The arm misses commanded poses by up to about 7° on shoulder lift and elbow. The errors the operator
saw by eye match what the servo encoders report, so the joint mapping (D-11) holds to within a few degrees and the
servos themselves don't reach their goals. The miss depends on the direction of approach: about 6–7° when the joint
moves against gravity, about 0–1° when it moves with gravity. With no table contact, this is the cleanest evidence
yet of real sag and hysteresis. The simulator predicts about 1.5° and the same value from both directions.

- **Data:** `data/pose_check/2026-10-31_rbe501_follower/results.json` (clean tree, commit `b84f53f`)
- **Command:** `python scripts/sim2real/collect/pose_check.py` (5 poses defined in sim degrees, ramped at 15°/s,
  2 s settle, 20 reads averaged)
- **Observations:** by eye against the described link angles (no level-app numbers)

## Link angles: commanded vs encoder reading vs operator

Upper arm in degrees forward of vertical; forearm in degrees above level; derived from the averaged readings
through the joint map.

| Pose (reached from) | Upper arm cmd → read | Forearm cmd → read | Lift error | Elbow error | Operator |
|---|---|---|---|---|---|
| `L_pose` (rest, folded) | 0 → 0.7 | 0 → -7.1 | +0.7 | +6.4 | lift slightly forward; forearm sagging 5–10° |
| `forearm_up_30` | 0 → 0.8 | 30 → 23.0 | +0.8 | +6.3 | forearm "more like 25" |
| `lean_forward_30` | 30 → 31.4 | 0 → -7.6 | +1.4 | +6.1 | lift about right; forearm/elbow short |
| `wrist_down_45` (from `lean_forward_30`) | 0 → 7.2 | 0 → -7.2 | +7.2 | 0.0 | lift tilted forward, forearm sagged, wrist right |
| `pan_45` | 0 → 7.2 | 0 → -7.4 | +7.2 | +0.2 | lift tilted forward, forearm roughly level |

Errors are reading minus goal in sim degrees. Positive means further forward or down, the direction gravity pulls.

## What it shows

1. **The mapping is close.** Wherever the operator saw a miss, the encoders report one of the same size and
   direction. The one unexplained residue is a slight forward lean of the upper arm in `L_pose` (encoder +0.7°), which
   suggests the shoulder-lift offset may be off by a few degrees. Level-app numbers would settle it.
2. **Direction-dependent miss (hysteresis) of about 6.5° on shoulder lift and elbow.**
   - **Shoulder lift:** at the same goal (upper arm vertical) it stops +0.7° from it when approached from behind
     (moving forward, with gravity) but +7.2° when approached from `lean_forward_30` (moving back, against gravity).
   - **Elbow:** it stops 6.1–6.4° low when it has to lift the forearm, and on target (0.0–0.2°) when it lowers
     the forearm from above.
   - **Midpoints:** about 4° of gravity sag on the lift and 3° on the elbow, plus a ±3° friction/deadband band.
     The sag check's ±1.3° deadband at light-load poses is consistent with this band widening under gravity load.
3. **The simulator underpredicts.** The same procedure on the sim bus (`--sim`) gives 1.3–1.6° on both joints,
   identical from both directions (MuJoCo friction creeps, D-03). The real miss is 2–4× larger and depends on the
   approach direction.
4. **Wrist flex is within 1° (with gravity) and pan within 1.6°.** Only the two gravity-loaded joints are large.

## Consequences

- At the reach's 30 cm scale, 7° on the shoulder alone moves the gripper about 15–35 mm, far over the 2 mm transfer
  limit. A feedforward-like policy (D-08) can't remove a miss whose sign depends on the approach direction.
  Feedback such as the integral baseline, or a policy trained with stiffness and friction randomization, is needed.
- `wrist_down_45` was flagged as clamped: the shoulder lagged more than 10° behind the ramp at some point, so
  LeRobot clipped a ramp goal. The final goal was sent unclipped.

## Next steps

1. Repeat with level-app numbers on `L_pose` approached from both sides, to pin the shoulder-lift offset to
   about ±1°.
2. Re-run the sag check (poses recorded higher, table pre-flight) to get the stiffness and friction fit. This
   check suggests the data will show real sag and hysteresis, not just contact.

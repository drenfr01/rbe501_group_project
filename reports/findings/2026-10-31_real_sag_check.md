# Finding: first real sag check on the SO-101 follower (2026-10-31)

**TL;DR:** The first session was compromised by table contact. Every pose was recorded with plenty of clearance
while held by hand. During the run, though, the gripper rested on the table at the horizontally reaching poses, then
hit the table and scraped along it as the arm retracted. Most shoulder-lift and elbow values therefore measure the
table, not the servos. One result survives: the poses without contact show every joint stopping about 1.3° short of
its goal from either direction (a deadband). The simulator does not model this. The stiffness measurement has to be
redone with a table check in place.

- **Session:** `data/sag_check/2026-10-31_rbe501_follower/` (`session.json`, `manifest.json`; clean tree at commit `e0b129e`)
- **Poses:** `data/sag_check/poses/poses.json`, 10 poses jogged by hand, each at least 5° inside the calibrated joint limits
- **Command:** `python scripts/sim2real/collect/sag_check.py run`

## What was measured

Each pose was approached twice: once with every arm joint starting 5° above the goal in LeRobot degrees, and once
starting 5° below. "Above" means the positive LeRobot direction, not physically up. After each step onto the pose
the arm settled for 2 s, then 20 position readings were averaged. LeRobot servo settings were the defaults on every
motor: P/I/D = 16/0/32, acceleration 254, torque limit 1000 (500 on the gripper).

**Sag = commanded minus measured, in LeRobot degrees** (from above / from below). Rows marked *contact* are
the reaching poses where the gripper rested on the table (operator observation). Rows marked *pinned* are poses
reached by retracting, where the gripper caught on the table.

| Pose | Contact | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll |
|---|---|---|---|---|---|---|
| 0 | none seen | -1.41 / 1.05 | -0.18 / 2.90 | -6.85 / -0.52 | -2.02 / 0.79 | -1.32 / 1.06 |
| 1 | none seen | -1.41 / 1.14 | -0.71 / -0.18 | -4.66 / -0.70 | -2.02 / 0.62 | -1.32 / 1.06 |
| 2 | contact | -1.41 / 1.23 | -4.49 / -4.05 | -5.19 / -2.29 | -2.11 / 0.44 | -1.32 / 1.06 |
| 3 | contact | -1.32 / 1.14 | -5.98 / -5.80 | -3.43 / -2.37 | -2.02 / 0.44 | -1.32 / 1.15 |
| 4 | contact | -1.41 / 1.23 | -5.80 / -5.36 | -7.03 / -2.99 | -0.88 / 0.97 | -1.32 / 1.06 |
| 5 | contact | -1.41 / 1.14 | -5.81 / -5.45 | -4.75 / -2.55 | -3.08 / 0.44 | -1.32 / 1.06 |
| 6 | contact | -1.49 / 1.06 | -5.80 / -5.45 | -9.84 / -5.54 | -0.97 / 0.88 | -1.41 / 1.14 |
| 7 | pinned | -1.32 / 1.32 | -17.58 / -13.89 | -2.38 / -1.50 | -1.41 / 0.97 | -3.43 / -1.32 |
| 8 | contact | -1.49 / 1.14 | -4.93 / -4.49 | -12.05 / -5.80 | -1.59 / 0.79 | -1.32 / 1.06 |
| 9 | pinned | -1.50 / 1.23 | -19.87 / -14.33 | -2.73 / -1.41 | -1.50 / 0.88 | -1.32 / 1.06 |

Motor status during the session: 23 to 32 °C, supply 5.3 to 5.5 V, highest load 38% of rated torque (shoulder lift).

## What went wrong: why the arm went lower than the recorded poses

The poses were recorded with the arm held by hand, so none of the effects below were visible then.

1. **Gravity sag.** Once the servos hold the arm instead of a hand, it droops. Extended horizontally, a few degrees
   at the shoulder and elbow lower the gripper by several centimetres.
2. **The approach offsets move every joint at once.** Shoulder, elbow and wrist flex were all shifted 5° in the same
   LeRobot direction, so the shifts add up at the gripper (5 cm or more). For some joints "above" moves the gripper
   toward the table.
3. **Moves between poses are straight lines in joint angles, not in space.** Every joint interpolates at the same
   time, and the gripper path between two safe poses can swing lower than either one. Retracting from a reaching
   pose (pose 6 to 7: shoulder lift 79° to 29°; pose 8 to 9: 58° to 14°) drove the gripper into the table, where it
   scraped. The ramp back to rest after pose 9 dragged along the table the same way, triggering LeRobot's
   "Relative goal position magnitude had to be clamped" warnings for the whole ramp.
4. **No table check was active.** The safety monitor that rejects commands near the table needs the joint mapping
   (Phase 1) to know where the gripper is, and the sag check ran before the mapping existed.

LeRobot never sends a goal more than 10° from the present position, which probably kept the contact force low.
Loads stayed at or below 38% and temperatures at or below 32 °C. The gripper and table surface should still be inspected.

## What the data still supports

### Deadband of about 1.3° on every joint (holds up)

At the poses without contact (0 and 1), and consistently at all other poses, shoulder pan and wrist roll (almost no
gravity load) land 1.3 to 1.4° above the goal when approaching from above, and 1.05 to 1.15° below it when
approaching from below. Wrist flex shows the same pattern around a small gravity offset (-2.0 / +0.6 to +0.8). The
20 readings at each pose were identical, so the joints were stuck, not jittering.

The likely cause is LeRobot's low gain (P = 16): near the goal the motor pushes too weakly to overcome gear friction.
This is a property of the servo combined with these settings, so a higher P gain may shrink it.

**Sim:** MuJoCo position actuators converge exactly to an unloaded goal, and the sim's friction model creeps back to
the same angle from both sides (`reports/differences.md`, D-03). At the gripper, ±1.3° on the shoulder and elbow is
roughly 5 to 10 mm, already larger than the policy's 0.3 mm sim error and the 2 mm transfer limit.

### Supply voltage of 5.4 V (holds up)

The servos read 5.3 to 5.5 V. That is typical for the standard SO-101 kit (a 5 V supply driving the 7.4 V STS3215
variant), and below the motors' rated voltage. Confirm the supply rating and the servo variant (`reports/differences.md`, D-09).

## What the data does not support

- **Stiffness and gravity sag:** the shoulder-lift and elbow values at poses 2 to 9 include table contact. The nearly
  identical shoulder-lift value (about -5.8°) across poses 3 to 6 is what a joint stopped by the table looks like, not
  a spring. No stiffness can be fitted from this session.
- **Elbow friction (direction-dependent offsets of 4 to 6°):** likely contaminated by contact; recheck.
- **Shoulder lift "unable to reach" poses 7 and 9:** caused by the gripper catching on the table, not motor weakness.
- **Poses 0 and 1** may be clean (no contact was seen), but two poses are too few for a fit.

## Next steps

1. **Phase 1 first:** run `mirror_arm.py` to get the joint signs and offsets, so the sim knows where the gripper is.
2. **Code fixes before any rerun:**
   - **Pre-flight check:** simulate every commanded point (the poses, both approach points, and the interpolated paths
     between poses and to rest), including an allowance for sag. Refuse the run if anything comes within the table
     clearance.
   - **Gentler approach:** offsets of about 3°, still larger than the 1.3° deadband, with each joint's direction
     chosen from the mapping so the gripper moves away from the table. Move between poses through a raised
     waypoint.
   - **Logging:** record the goal LeRobot actually sends, and flag any sample where it was clamped.
3. **Re-record poses** with extra height, expecting the arm to sit several centimetres lower once the servos hold
   it, then rerun the sag check.

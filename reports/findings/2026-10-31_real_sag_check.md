# Finding: first real sag check on the SO-101 follower (2026-10-31)

**TL;DR:** The real servos don't behave like the spring the simulator assumes. Every joint stops about 1.3° short
of its goal from whichever side it approaches (a deadband). The elbow lands in a different place depending on the
approach direction (friction). Shoulder lift couldn't reach two poses at all. The arm also runs at 5.4 V. The
simulator's position actuators model none of this.

- **Session:** `data/sag_check/2026-10-31_rbe501_follower/` (`session.json`, `manifest.json`; clean tree at commit `e0b129e`)
- **Poses:** `data/sag_check/poses/poses.json`, 10 poses jogged by hand, each at least 5° inside the calibrated joint limits
- **Command:** `python scripts/sim2real/collect/sag_check.py run`

## What was measured

Each pose was approached twice: once with every arm joint starting 5° above the goal, and once starting 5° below.
After each step onto the pose the arm settled for 2 s, then 20 position readings were averaged. LeRobot servo
settings were the defaults on every motor: P/I/D = 16/0/32, acceleration 254, torque limit 1000 (500 on the
gripper).

**Sag = commanded minus measured, in LeRobot degrees** (from above / from below):

| Pose | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll |
|---|---|---|---|---|---|
| 0 | -1.41 / 1.05 | -0.18 / 2.90 | -6.85 / -0.52 | -2.02 / 0.79 | -1.32 / 1.06 |
| 1 | -1.41 / 1.14 | -0.71 / -0.18 | -4.66 / -0.70 | -2.02 / 0.62 | -1.32 / 1.06 |
| 2 | -1.41 / 1.23 | -4.49 / -4.05 | -5.19 / -2.29 | -2.11 / 0.44 | -1.32 / 1.06 |
| 3 | -1.32 / 1.14 | -5.98 / -5.80 | -3.43 / -2.37 | -2.02 / 0.44 | -1.32 / 1.15 |
| 4 | -1.41 / 1.23 | -5.80 / -5.36 | -7.03 / -2.99 | -0.88 / 0.97 | -1.32 / 1.06 |
| 5 | -1.41 / 1.14 | -5.81 / -5.45 | -4.75 / -2.55 | -3.08 / 0.44 | -1.32 / 1.06 |
| 6 | -1.49 / 1.06 | -5.80 / -5.45 | -9.84 / -5.54 | -0.97 / 0.88 | -1.41 / 1.14 |
| 7 | -1.32 / 1.32 | **-17.58 / -13.89** | -2.38 / -1.50 | -1.41 / 0.97 | -3.43 / -1.32 |
| 8 | -1.49 / 1.14 | -4.93 / -4.49 | -12.05 / -5.80 | -1.59 / 0.79 | -1.32 / 1.06 |
| 9 | -1.50 / 1.23 | **-19.87 / -14.33** | -2.73 / -1.41 | -1.50 / 0.88 | -1.32 / 1.06 |

Motor status during the session: 23 to 32 °C, supply 5.3 to 5.5 V, highest load 38% of rated torque (shoulder lift).
Nothing came close to overload.

## Findings

### 1. Every joint stops about 1.3° short of its goal (a deadband)

Shoulder pan and wrist roll carry almost no gravity load. Even so, they land 1.3 to 1.5° above the goal when
approaching from above, and 1.1 to 1.3° below it when approaching from below, at every pose. That's about 15 encoder
counts. The 20 readings at each pose were identical (zero spread), so the joint was stuck, not jittering.

The likely cause is LeRobot's low gain (P = 16). Near the goal the motor's push is too weak to overcome gear
friction, so the joint stops wherever that happens.

**Sim:** MuJoCo position actuators converge exactly to the goal (zero error with no load). The sim's friction model
creeps back to the same angle from both sides within about 2 s, so it can't reproduce this either
(`reports/differences.md`, D-03).

### 2. The elbow lands in a different place depending on the approach direction

The elbow's result depends on which side it came from by 4 to 6°. For example, pose 8 reads -12.05° from above and
-5.80° from below, and pose 0 reads -6.85° and -0.52°. That is far more friction than the deadband on the unloaded
joints, and it changes from pose to pose.

### 3. Shoulder lift couldn't reach poses 7 and 9

Shoulder lift ended 14 to 20° away from its goal at poses 7 and 9, both of which have a small shoulder-lift angle
(28.8° and 14.0°). LeRobot never sends a goal more than 10° away from the current position (`max_relative_target`).
When the joint fell that far behind, LeRobot silently moved the goal closer, so the goal actually sent was not the
one recorded.

The same thing happened on the ramp back to rest: shoulder lift trailed the command by 10 to 18° the whole way,
triggering a run of "Relative goal position magnitude had to be clamped" warnings. Something holds the shoulder up
in that range of angles: a weak motor, a cable, or contact. This still needs to be checked by looking at the arm.

**These two samples are not valid sag measurements.** The sag-check script does not yet log the goal LeRobot
actually sent; that is a code fix to make.

### 4. The supply is 5.4 V

The servos read 5.3 to 5.5 V. That is typical for the standard SO-101 kit (a 5 V supply driving the 7.4 V STS3215
variant), but the motors are weaker than their rating, which makes every sag larger. Confirm the supply rating and
the servo variant before comparing with other arms (`reports/differences.md`, D-09).

### 5. Other observations

- Shoulder lift shows nearly the same sag (about -5.8° from above) at poses 3 to 6. That fits either similar gravity
  torque at those poses or a motor at its limit. The two can't be told apart until the joint mapping gives the
  gravity torque per pose.
- Shoulder-lift sag is negative at almost every pose (the joint stops past the goal). Whether that is "down" depends
  on LeRobot's sign convention, which the mirror tool will establish.

## What this means for sim2real

- The simulator's model is "sag = gravity torque / stiffness", with no deadband and no lasting friction. The real
  error is dominated by a deadband of about ±1.3° on every joint, plus direction-dependent friction on the elbow.
  At the gripper, ±1.3° on the shoulder and elbow is roughly 5 to 10 mm. That is already larger than the policy's
  0.3 mm sim error and the 2 mm transfer limit.
- This supports the Phase 2 finding that only feedback control is robust. Integral control keeps pushing until the
  joint actually arrives, which is exactly what a deadband defeats in a plain position controller. The learned
  policy behaved like feedforward in sim, so it is unlikely to fix a deadband it never saw in training.
- The stiffness fit is not run yet. It needs the joint mapping (signs and offsets) to compute gravity torque at each
  pose, and should exclude poses 7 and 9.

## Next steps

1. **Code:**
   - log the goal actually sent and flag clamped samples;
   - exclude flagged samples from the fit;
   - report the deadband per joint as its own number, alongside stiffness.
2. **Hardware:** check what limits shoulder lift at poses 7 and 9. Confirm the power supply rating and the servo variant.
3. **Phase 1:** run `mirror_arm.py` for the joint mapping, then `analyze_sag.py` on this session.
4. **Sim:** consider modeling the deadband (a backend option that leaves the error uncorrected below a threshold) and
   include it in the robustness sweep and any retraining.

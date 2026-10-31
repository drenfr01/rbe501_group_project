# Session recap: first hardware day on the SO-101 follower (2026-10-31)

**TL;DR:** The real arm doesn't behave like the simulator. Under gravity, the shoulder-lift and elbow servos stop
up to 7° short of their goals, and how far short depends on which way they were moving (about 6.5° of hysteresis).
The sim predicts 1.5° from both sides. Even the 1.3° deadband on unloaded joints is enough to miss the 2 mm reach
tolerance. The sim-only robustness sweep had already shown the trained reach policy is fragile to these effects, so
our plan is to retrain with randomized dynamics, or compare against feedback control, rather than transfer the
current policy unchanged. We now have a working joint mapping between LeRobot and the MJCF, plus tools to measure
the arm safely. The first sag check was compromised by table contact and has to be redone.

Detailed notes, one per experiment:

- [Real sag check](2026-10-31_real_sag_check.md)
- [Pose check](2026-10-31_pose_check.md)
- [Differences log D-01 to D-11](../differences.md)
- [Assumptions table](../assumptions.md)

## Where we started

- A residual-IK reach policy (`checkpoints/reach_1m_residual_ik`) that holds 0.34 mm in sim, trained without
  dynamics randomization.
- A provisional sim-only sweep (`experiments/2026-10-25_reach_robustness/`) with the verdict **retrain**:
  - **Stiffness:** the policy meets the 2 mm limit only for actuator stiffness between 0.75× and 1.5× nominal
    (22 mm at 0.3×).
  - **Timing:** encoder rounding plus a one-step delay raises its error to 14.35 mm, worse than plain IK hold.
  - **Type of correction:** the policy behaves like feedforward (a fixed correction tuned to the nominal plant).
    Joint integral control stays at 0.77 mm everywhere.
- No real measurement of stiffness, friction, joint offsets, or reachable range.

## What we did today, in order

### 1. Sag check, first attempt: compromised by table contact

10 poses recorded by hand, then each approached from 5° above and below with torque on.

- **Contact:** the gripper hit the table at the reaching poses and scraped along it while retracting.
  - **Causes:** gravity sag; the "above/below" offsets moved every joint the same way, which lowered the gripper
    on one side; and joint-space interpolation dipped the gripper.
  - **Why nothing stopped it:** no table check could run, because the joint mapping didn't exist yet.
- **Still valid:** about ±1.3° deadband on every joint at light load (each joint stalls just short of its goal
  from either direction). Supply voltage is 5.3–5.5 V.
- **Not valid:** stiffness and friction values at poses 2–9.
- **Revisited later:** with the mapping fitted (step 3), the measured poses sit 40–140 mm above the sim table. So the
  contact most likely happened while moving between poses, and part of the large errors may be real sag.

### 2. Logging fix: record what LeRobot actually sends

LeRobot clips every goal to within 10° (`max_relative_target`) of the present position. The sag check now records
the goal LeRobot actually sent and flags each sample where it was clipped. `analyze_sag.py` excludes flagged samples
and reports how many it dropped.

### 3. Joint mapping (Phase 1): LeRobot degrees vs MJCF

8 torque-off references with `mirror_arm.py` (upper arm vertical or flat, forearm vertical or flat, wrist poses).

- **Fit:** `sim = lerobot + offset` with every sign +1.
  - pan +1.5°
  - shoulder lift −26.8°
  - elbow −6.7°
  - wrist flex −0.5°
  - wrist roll 0
- **Why it mattered:** without the offsets, the sim drew poses that were in the air as 1–64 mm inside the table.
  That was the "shoulder and elbow tilted down" seen in the viewer.
- **Rejected alternatives:**
  - Flipping the shoulder sign puts a hand-held reference 165 mm inside the table.
  - Fitting to the line between the joint axes instead of the visible links is 14° off.
- **Uncertainty:** three "upper arm flat" references disagree by 8–14°, and the elbow references scatter ±7°.
- **Reachable range** (sim degrees): the real shoulder reaches lift −115 to +61 against the model's ±100. So it
  goes further back but 39° less far forward.
  - 4.8% of training targets need a joint angle the hardware can't reach (mostly elbow).
  - All 10 fixed trial targets are reachable.

### 4. Pose check: command known poses, compare encoders and eyes

Five poses defined in sim degrees: L pose, forearm up 30°, lean forward 30°, wrist down 45°, pan 45°. The whole
path was pre-checked against the table with a sag allowance. Ramped at 15°/s.

- **Mapping:** close. Every miss seen by eye showed up in the encoder readings with the same size and direction.
  The shoulder offset may still be off by a couple of degrees.
- **Servos:** shoulder lift and elbow stop 6–7° short when moving against gravity and 0–1° short when moving with
  it. Wrist and pan are within about 1.6°.
- **Sim:** the same procedure gives 1.3–1.6° from both directions. The real error is 2–4× larger, and its
  direction dependence is something MuJoCo's friction model can't show (D-03).

## What we understand now

1. **Servos are softer than the model and sticky.** At LeRobot's default P = 16:
   - **Sag:** about 3–4° of gravity sag on shoulder lift and elbow at the L pose, against about 1.5° in sim. That
     suggests the real stiffness is 2–3× lower.
   - **Friction:** a ±1.3° deadband unloaded, widening to about ±3° under gravity load.
   - **Scale:** at reach scale, 7° at the shoulder is 15–35 mm at the gripper.
2. **The current policy won't transfer.** Its sim-validated stiffness window is 0.75–1.5× nominal. The real arm is
   likely below that, and the miss changes sign with the direction of approach, which no fixed correction can
   cancel. This confirms the provisional **retrain** verdict.
3. **Encoders and external references answer different questions.** The encoders show how far each servo is from
   its goal. A level or by-eye reference is only needed to check the mapping from encoder angle to physical link
   angle.
4. **Safety needs the mapping.** The table check is only as good as the joint offsets. All powered tools now
   pre-check their full path.

## Tools built today

| Tool | Purpose |
|---|---|
| `collect/sag_check.py` | sag and hysteresis per pose; logs the goal LeRobot actually sent and flags clipped samples |
| `collect/mirror_arm.py` | torque-off live mirror in MuJoCo; records reference poses |
| `collect/pose_check.py` | sends the arm to sim-defined poses after a full-path table check; records goal, sent, read, and operator measurement |
| `HardwareConfig.joint_map` | fitted offsets, with tests that hand-held poses map above the table and vertical references draw vertical |

All scripts refuse to run on a dirty tree, write a manifest, and support `--sim` dry runs where it makes sense.

## Open questions

- **Shoulder-lift offset:** by-eye accuracy is about ±3°. Is it within 1°?
- **Contact in the first sag session:** did it happen at the held poses or only during moves? This decides how
  much of that data is usable.
- **Gain:** does a higher P gain shrink the deadband and sag enough to matter, and should the real runs use it?
  If so, the sim must model the new gain.
- **Servo variant:** 7.4 V or 12 V STS3215? It affects the stall torque and therefore the stiffness (D-09).

## Where we're going

1. **Pin the mapping.** Run `pose_check.py --poses L_pose lean_forward_30 L_pose` with level-app numbers, and
   update the shoulder and elbow offsets if they're off by more than about 1°.
2. **Redo the sag check properly.**
   - **Code first:** a table pre-flight for the sag procedure, gentler approach offsets (about 3°) in directions
     that move the gripper away from the table, and a raised waypoint between poses.
   - **Then:** re-record the poses higher, run, and use `analyze_sag.py` to write
     `data/calibration/actuator_calibration.json` (stiffness and friction per joint).
3. **Calibrated sweep.** Run `robustness_sweep.py --calibration ... --session ...` for the formal transfer-or-retrain
   decision, now with the real stiffness.
4. **Retrain** (`residual_ik_calibrated`) with:
   - the calibrated stiffness;
   - stiffness and friction randomization;
   - a 0–1 step delay;
   - targets restricted to the real reachable range.

   A deadband or stiction model may be needed in sim, because MuJoCo's friction creeps (D-03).
5. **Phase 3 hardware trials.** Run `run_policy_real.py` for IK hold, gravity compensation, integral, and the
   retrained policy on `reach_fixed.json`, then 10 repeats on `reach_repeat.json`, each with a calibrated-sim
   prediction first. Set `HardwareConfig.rest_deg` to a supported pose before this step.

## For the paper

The story so far: a policy with 0.34 mm error in sim meets an actuator whose sag is 2–4× larger than modeled and
whose friction gives about 6.5° of direction-dependent error. The simulator's friction model can't reproduce that
hysteresis at all. The sim-only sweep predicted this fragility before we touched hardware, and the hardware data
confirmed it. The evidence for each claim is in `differences.md` (D-01, D-03, D-08, D-11) and in the findings notes
linked above.

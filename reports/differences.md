# Sim-vs-real differences log

One entry per difference, with a stable ID the report text can cite. "Status: open" means the real-arm
measurement has not been made yet; those entries state what the sim side is and how it will be measured.
Spec: [`docs/sim2real_reach_transfer_spec.md`](../docs/sim2real_reach_transfer_spec.md).

### D-01: Actuator stiffness (kp) is a placeholder

- **Sub-task:** reach, Phase 0 (real sag check) and Phase 2 (stiffness sweep)
- **Sim:** MJCF position actuators kp = 20 / 20 / 20 / 6 / 4 / 4 N·m/rad (pan, lift, elbow, wrist flex, wrist roll,
  gripper) with `dampratio=1`. None were measured.
- **Real:** LeRobot writes P/I/D = 16/0/32 to every STS3215 on connect; the resulting stiffness in N·m/rad is
  unknown until `sag_check.py run` + `analyze_sag.py`. The pose check gives a first estimate: about 3–4° of gravity sag
  on shoulder lift and elbow at the L pose (midpoint of the two approach directions), versus about 1.5° in sim.
  That suggests the real stiffness is 2–3× lower than the MJCF value, before the deadband is accounted for.
- **Impact (sim, measured):** the residual-IK policy holds 0.34 mm at nominal kp but 6.3 mm at 0.5x and 22.4 mm at
  0.3x; it meets the 2 mm transfer limit only for kp scales 0.75 to 1.5 (D-08, A-01).
- **Action:** `ActuatorCalibration` file (`data/calibration/actuator_calibration.json`) loaded on top of the MJCF;
  `ActuatorCalibration.apply` keeps the damping ratio (kv scaled by sqrt(kp_new / kp_old)).
- **Evidence:** [Phase 2 sweep](../experiments/2026-10-25_reach_robustness/README.md), [kp_sweep.png](figures/kp_sweep.png)

### D-02: Joint ranges, and training targets the real arm cannot reach

- **Sub-task:** reach, Phase 1 (joint mapping) and Phase 3 target selection
- **Sim:** MJCF ranges (deg): pan ±110, lift ±100, elbow ±96.8, wrist flex ±95, wrist roll -157 to 163,
  gripper -10 to 100.
- **Real:** LeRobot calibration `rbe501_follower` (half range in LeRobot degrees): pan ±118.7, lift ±87.8,
  elbow ±97.0, wrist flex ±104.9, wrist roll full turn. The real backend clips to the tighter of both minus
  `limit_margin_rad` (0.15 rad = 8.6 deg), so shoulder lift is limited to ±79.2 deg on hardware versus ±100 in training.
- **Impact (measured):** under the Phase 1 mapping (D-11) the real ranges in sim degrees are lift -115 to +61 and
  elbow -104 to +90, so the real shoulder reaches 15° further back than the model but 39° less far forward. With the
  margin the hardware window is lift -91 to +52, elbow -88 to +82. Of 400 sampler configurations, 19 (4.8%) fall
  outside it (elbow 15, lift 5). Under the earlier identity mapping the estimate was 38 (9.5%), all shoulder lift. On
  the sim bus such a target leaves every controller, including integral control, about 31 mm off because the command
  saturates at the clip. All 10 targets in `reach_fixed.json` stay inside the window under the new mapping.
- **Action:** `data/targets/reach_fixed.json` keeps only targets whose IK pose is inside the hardware range minus the
  margin (3 of the first 13 rejected). A retrained policy should sample targets within the real range.
- **Evidence:** `run_policy_real.py --sim` on the unfiltered first target; the target files' `source` field.

### D-03: MuJoCo friction creeps, so the sim has no lasting hysteresis

- **Sub-task:** reach, Phase 0 (hysteresis) and the calibrated-sim check
- **Sim (measured):** with `dof_frictionloss` = 0.15 N·m on shoulder lift / elbow / wrist flex (kp 20 / 20 / 6), the sag
  procedure (approach 5 deg from above and from below) gives hysteresis 0.3 deg on some joints after 0.05 s of settling,
  about 0.1 deg after 0.3 s, and below 0.0001 deg after the procedure's 2 s settle. A Coulomb joint would hold about
  frictionloss / kp = 0.43 deg (lift, elbow) indefinitely. MuJoCo's friction loss is a soft constraint and the joint
  slowly creeps to the same equilibrium from both sides.
- **Real (measured, pose check):** the final position depends on the approach direction by about 6.5° on shoulder lift
  and elbow under gravity load: +7.2° vs +0.7° (lift) and +6.4° vs 0.0° (elbow) at the same goals. The sim gives
  1.3–1.6° from both sides. See [the pose-check finding](findings/2026-10-31_pose_check.md).
- **Impact:** the calibration's `frictionloss` (from `friction_torque = kp * hysteresis`) changes the transient but
  cannot reproduce a real steady-state hysteresis. The calibrated-sim check can match sag but not hysteresis.
- **Action:** report real hysteresis separately as an unmodeled effect; only write friction when hysteresis exceeds
  one encoder count. A stiction model (or a deadband in the backend) is needed if hysteresis turns out large.

### D-04: Encoder quantization

- **Sub-task:** reach, all phases
- **Sim:** continuous joint angles and physics velocities.
- **Real:** STS3215 encoders have 4096 counts per turn (0.088 deg). LeRobot sends goals as whole counts, so commanded
  angles, and therefore measured sag, come in 0.088 deg steps. Velocity has to be differenced from rounded positions.
- **Impact (measured, sim):** encoder rounding alone raises the policy's mean holding error from 0.34 to 1.00 mm and its
  action saturation from 18% to 19%; the baselines barely change (IK hold 10.29 to 10.37 mm, integral 0.77 to 0.85 mm).
  In the stiffness fit, poses at whole degrees with similar sag made the rounding errors correlated and biased kp by
  about 1.2% with a too-narrow CI.
- **Action:** `PerturbedBackend` rounds positions and differences velocity like the real backend; the fit averages
  per pose and test poses are jittered off whole degrees.

### D-05: Command-to-measurement delay

- **Sub-task:** reach, Phase 2 (sensing and timing) and Phase 3 (loop overruns)
- **Sim:** the command applied in a step is seen in the same step's observation.
- **Real:** each control step makes several bus transactions (`send_action` adds its own position read), so at least
  one 33 ms step of delay is plausible. Overruns are counted per session by `LeRobotSo101Backend`.
- **Impact (measured, sim):** with rounding plus a one-step delay the policy's error rises to 14.35 mm (seeds
  12.3 / 16.0 / 14.7), worse than IK hold (10.37 mm), with 38.6% of holding steps saturated. Gravity compensation
  (0.81 mm) and integral control (0.85 mm) are unaffected because they converge slowly.
- **Action:** the decision rule uses this condition; `residual_ik_calibrated` trains with a random 0 to 1 step delay.

### D-06: Start state

- **Sub-task:** reach, Phase 3
- **Sim:** reset teleports the arm to home at rest.
- **Real:** the backend holds the present pose (present position written as the goal before torque is enabled),
  ramps to home over `home_ramp_s`, then waits `settle_s`. The arm starts each episode sagged and possibly with
  residual motion.
- **Impact:** not yet measured on hardware. Driving the sim arm through the real backend (ramped start, half speed,
  rounding, the hardware-reachable target set) the baselines hold 10.0 mm (IK hold) and 0.82 mm (integral), in line
  with the teleported-start sweep (10.4 and 0.85 mm with rounding). Different targets, so not a controlled comparison.

### D-07: Command speed

- **Sub-task:** reach, Phase 2 (reduced speed) and Phase 3
- **Sim:** training rate limit 0.07 rad per 33 ms step.
- **Real:** first hardware runs use `speed_scale` 0.5.
- **Impact (measured, sim, with rounding and one-step delay):** the policy's error drops from 14.35 to 10.61 mm at half
  speed; the slow approach still settles before the holding window for the baselines.

### D-08: The learned correction is feedforward-like

- **Sub-task:** reach, Phase 2
- **Sim (measured):** over kp scales 0.3 to 3 the policy's worst error is 65x its nominal error; gravity
  compensation (pure feedforward) grows 32x and integral control (pure feedback) 1.0x. The policy is classified
  feedforward: it learned a correction tuned to the nominal plant rather than closing the loop on the measured error.
  It was trained without dynamics randomization.
- **Impact:** transfer depends on the sim matching the real stiffness to within about -25% / +50%.
- **Action:** the provisional verdict is retrain; see [assumptions.md](assumptions.md).

### D-09: Servo variant and supply voltage

- **Sub-task:** reach, Phase 0
- **Sim:** one torque model for all joints.
- **Real:** open. STS3215 ships in 7.4 V and 12 V variants with different stall torque; the voltage read during the sag
  session (`MotorStatus.voltage_v`) and the variant must be recorded before comparing stiffness across arms.

### D-10: Gripper torque limit

- **Sub-task:** deferred to the cube task
- **Sim:** gripper actuator with the MJCF force range.
- **Real:** LeRobot writes `Max_Torque_Limit` = 500 (50%) to the gripper on connect. Irrelevant for a bare-arm reach,
  but it bounds grip force for the cube task.

### D-11: Joint zero offsets between LeRobot calibration and the MJCF

- **Sub-task:** reach, Phase 1 (joint mapping)
- **Sim:** MJCF zero pose draws the upper arm vertical and the forearm and gripper horizontal. The line between the
  shoulder-lift and elbow axes is 14° off the drawn upper arm, so "vertical" must be judged on the mesh, not the axes.
- **Real:** LeRobot DEGREES puts 0 at the midpoint of each calibrated range, which is not the MJCF zero. Eight
  references were taken with `mirror_arm.py` (torque off, link orientations judged by eye):
  `data/joint_mapping/2026-10-31_rbe501_follower/references.json`.
- **Fit (`sim = lerobot + offset`, all signs +1):** pan +1.5° (three "straight ahead" references), shoulder lift
  -26.8° (two "upper arm vertical" references, 27.7 and 26.5, plus one "flat" reference, 26.2 from the 90° mark),
  elbow -6.7° (mean of four 90° bends, spread ±7°), wrist flex -0.5° (two 90° references, 0.2° apart). Wrist roll
  was not referenced (0).
- **Consistency:** three "upper arm flat" references with the elbow folded or the wrist turned disagree by 8 to 14°
  with the vertical ones, more than eye judgment of a vertical link. Two candidate fits were rejected: the opposite
  shoulder sign puts a hand-held reference 165 mm inside the table, and an axis-to-axis fit (lift -41°) contradicts
  the renders. Under the identity mapping, sag poses 3 to 8 (held in the air when recorded) were 1 to 64 mm inside
  the sim table; this is the "shoulder and elbow tilted down" seen in the viewer.
- **Impact:** every hardware sim-to-real comparison depends on this map: the safety monitor's table check,
  joint-limit clipping (D-02), gravity torque in the sag fit, and policy observations.
- **Action:** `HardwareConfig.joint_map` holds the fit; `tests/test_joint_mapping.py` checks the hand-held references
  map above the table and the vertical references draw a vertical upper arm. `pose_check.py` commands known sim
  poses on the powered arm for a level-app check (that check also includes servo sag). Still to do: a contact check
  (gripper resting on the table must read about 0 mm clearance in the mirror tool) and tighten the elbow and the
  "flat" references with a phone level.
- **Evidence:** [joint_mapping_references.png](figures/joint_mapping_references.png) (references 1, 2, 3, 5, 7
  rendered with the fit), [joint_mapping_identity.png](figures/joint_mapping_identity.png) (sim zero and references
  2, 1, 3 before the fit).

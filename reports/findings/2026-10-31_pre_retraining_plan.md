# Plan: everything to do before the retraining run (2026-10-31)

**Goal of the retrain:** a reach policy that meets the 2 mm limit on the real arm, unlike the current one. The current
policy only works within 0.75–1.5× the sim's stiffness and learned a fixed correction, while the real arm is softer
and has about 6.5° of direction-dependent error (see the [session recap](2026-10-31_session_recap.md)). Training
can only fix what the simulator reproduces. So most of this plan is about making the sim match the real arm and
showing that it does, before spending compute.

Status: **have** = built and tested; **new** = needs code; **run** = needs a hardware or sim session.

## A. Settle the hardware configuration (everything after this depends on it)

1. **Choose the servo P gain** (done: P = 24, see [gain sweep](2026-10-31_gain_sweep.md)). Run `pose_check.py --p-gain G --poses L_pose lean_forward_30 L_pose` for
   G = 16, 24, 32 (48 if 32 is clean). Pick the highest gain with no buzzing or wobble and acceptable temperature and
   load. Set it as `HardwareConfig.position_p_gain`. Every later calibration is specific to this gain.
   - **have:** `--p-gain` on `pose_check.py`, `sag_check.py`, and `run_policy_real.py`.
   - **have:** each pose-check run gets its own folder `data/pose_check/<date>_<robot>_p<gain>_<time>/`, with the
     servo registers read back into `servo_settings.json`.
2. **Pin the joint offsets** (deferred: no level available, and by-eye isn't better than the current fit). A
   constant offset error of a few degrees affects sim and real alike through the same map. So it barely changes the
   sag fit, the calibrated-sim gate, or the encoder-based reach error; it matters only for external ground truth.
   Cheap upgrade when convenient: contact references in `mirror_arm.py` (forearm laid flat on the table, upper arm
   pressed against the side of a box) or a square-on side photo. Original plan (run): At the chosen gain, record level-app numbers on `L_pose` approached from both
   sides. Take the midpoint of the two encoder readings as the physical angle, and correct the shoulder-lift and
   elbow offsets in `HardwareConfig.joint_map` if they're off by more than 1°.
3. **Record the servo variant and supply** (done: 5 V kit supply, 7.4 V STS3215, see D-09) (7.4 V or 12 V STS3215, supply rating). This bounds the stall torque
   (D-09).
4. **Set `HardwareConfig.rest_deg`** (done: the folded supported pose read on 2026-10-31, 99 mm above the table clearance) to a pose where the arm is supported. It's where every run ramps to before
   torque off.

## B. Measure the actuators properly

5. **Make the sag check safe and non-destructive** (new):
   - table pre-flight over the poses, both approach points, and every interpolated path, with a sag allowance
     (reuse `preflight` from the pose check);
   - approach offsets of about 3°, signed per joint so the gripper moves away from the table;
   - a raised waypoint between poses;
   - a unique session folder per run (today's folder name would be overwritten by a second run today).
6. **Re-record the sag poses higher, then run the sag check** at the chosen gain (run).
7. **Fit the calibration** (have): `analyze_sag.py` writes `data/calibration/actuator_calibration.json` (stiffness
   per joint, friction where the hysteresis is identifiable).

## C. Make the simulator reproduce the real arm

8. **Deadband / stiction model** (new, the biggest gap). MuJoCo's `frictionloss` creeps to the same equilibrium
   from both sides (D-03), so the sim can't show the direction-dependent error. Add a backend wrapper, alongside
   `PerturbedBackend`, that holds the joint still while the commanded error is inside a band, and that band widens
   with gravity load. Fit its parameters to the sag-check and pose-check data.
9. **Model LeRobot's 10° relative-goal clip and the hardware speed** (new). Training applies neither, but on the
   real arm a lagging, sagged joint makes the clip bite (`wrist_down_45` was clipped today). Train at the hardware
   speed (`speed_scale` 0.5) or randomize it.
10. **Gate: the calibrated sim must reproduce the real data** (new + run). Replay today's pose check and the new sag
    check on the sim bus with the calibration and deadband model. Then compare per joint and per approach
    direction. Pass if the errors match within about 1° (enough to reach the 2 mm scale at the gripper).
    - This needs `--calibration` on `pose_check.py --sim`.
    - Report the result in `differences.md`; it's a central table for the paper.
    - Don't train until this passes. A sim that can't reproduce the error can't teach the policy to correct it.

## D. Configure the training run

11. **Restrict targets to the hardware window** (new). The sampler draws from the full MJCF range, and 4.8% of its
    configurations are unreachable on the real arm (D-02). Sample inside the real calibrated range mapped through
    `joint_map`, minus `limit_margin_rad`.
12. **Randomization centred on the calibration** (partly have). Today's ranges are kp 0.7–1.3×, damping 0.7–1.3×,
    payload 0–50 g, and delay 0–1 step. Re-centre kp on the calibrated value and widen it to cover the estimate's
    uncertainty and gain drift (the Phase 2 sweep suggests about 0.5–2×). Add randomization of the deadband/friction
    width (new). Keep encoder rounding.
13. **Observation choices.** The policy sees encoder positions, which include the sag, the same signal as on
    hardware. Check that the residual-IK action and observation layout have nothing that only exists in sim.
14. **Baselines and the bar to beat** (have the sweep, run it). Run `robustness_sweep.py --calibration ...
    --session ...` for the current policy and IK hold, gravity compensation, and integral control in the calibrated
    sim. Integral control is the bar: a retrained policy that doesn't beat it under randomization isn't worth
    deploying.
15. **Acceptance criteria, written down before training.** Proposed, all in the calibrated sim with randomization:
    - mean holding error under 2 mm, and worst-case kp scale under 5 mm;
    - no worse than integral control in any condition, and better on settling time;
    - saturation under 20%;
    - error after approaching from either direction within 1 mm of each other (no hysteresis left).
16. **Smoke test** (run). Train for about 50k steps on the final preset to check it learns and nothing crashes, and
    check the episode time per step for the compute estimate.
17. **Freeze and record.** Commit everything, train from a clean tree (the manifest records commit and versions),
    and use 3 seeds and the same step budget as `reach_1m_residual_ik` so the comparison is fair.

## E. After training (not part of the pre-flight, listed for the order)

18. Calibrated-sim sweep of the retrained policy against the acceptance criteria.
19. `run_policy_real.py --sim --calibration ...` for each controller, to get the predicted hardware numbers.
20. Phase 3 on hardware: `reach_fixed.json`, then 10 repeats on `reach_repeat.json`, for IK hold, gravity
    compensation, integral, and the retrained policy. Compare against the predictions; the gap is the paper's main
    sim-to-real result.

## Critical path

1 → 2 → 5 → 6 → 7 → 8 → 10 (gate) → 11/12 → 14 → 16 → train.

Items 3, 4, 9, 13 and 15 can be done in parallel. New code, in suggested order: 5 (sag-check safety), 10's
`--calibration` for the pose check, 8 (deadband model), 11 (target restriction), 9 (relative clip and speed), and
12's friction randomization.

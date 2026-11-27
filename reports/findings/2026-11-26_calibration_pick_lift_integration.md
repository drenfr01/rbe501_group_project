# Reaching calibration and pick-and-lift integration (2026-11-26)

## Scope

Integrate `transferring-trained-move-policy` at `b3d1814` with main at `4a73f52`.
Resolve configuration additions by retaining both tasks and resolve training
imports by retaining the calibrated reach factory and periodic/resumable
pick-and-lift checkpoints. Source MJCF, trained artifacts, and historic results
are unchanged. No new policy training or physical-robot operation was performed.

## Compatibility fixes

- Defer POSIX terminal imports until physical terminal input is opened. On
  Windows, actual terminal input reports a Linux/devcontainer requirement;
  simulation and scripted-key imports remain usable.
- Serialize provenance script paths with forward slashes on every platform.
- Skip the LeRobot configuration compatibility test explicitly when that optional
  hardware library is absent; retain the fake-bus and safety tests.

Initial combined tests exposed all three issues. After fixes and the additional
compiled-grasp-model calibration regression test: **356 passed, 1 skipped** on
Windows/Python 3.12.15. The skip is the real LeRobot configuration check.
CUDA remains available on the RTX 5070 with PyTorch 2.14.1+cu130.

## Which dynamics are actually applied

`residual_ik_sim2real_smooth` constructs a `PerturbedBackend` around a
`ServoEffectsBackend` around the measured-stiffness MuJoCo backend. A reset and
five-action smoke test produced finite 34-value observations.

Default pick-and-lift retains its original gains and contact settings. Explicit
`PickLiftConfig.sim.calibration_path` applies the measured arm stiffness and
rescales actuator velocity gains to preserve damping ratio. A regression test
checks that this survives reset without losing the compiled fingertip pads,
cube size, or object mass.

That option alone does **not** reproduce the reaching stiction, command clipping,
delay, quantization, or real joint-window wrappers. The calibration file covers
five arm joints, not moving-gripper dynamics; two arm stiffness values are
marked assumed. Contact friction remains an independent modeling assumption.

## Small checkpoint regression comparison

Headless rollouts, placement seeds 10000, 10001, 10002, fixed 10 g mass, 25 mm
cube, 200 steps per episode. Policies were loaded with their existing frozen
normalizers and run deterministically. The two trained policies are the
original seed-0 shared checkpoints, not newly trained models.

| Dynamics | Controller | Success | Final lifts (cm) |
|---|---|---|---|
| Original | Scripted | 3/3 | 6.49, 6.57, 6.53 |
| Original | Fixed-10-g SAC | 3/3 | 7.38, 7.05, 7.79 |
| Original | Random-5–20-g SAC | 3/3 | 6.91, 7.18, 6.64 |
| Measured stiffness only | Scripted | 3/3 | 6.24, 6.20, 6.13 |
| Measured stiffness only | Fixed-10-g SAC | 3/3 | 6.27, 6.20, 6.56 |
| Measured stiffness only | Random-5–20-g SAC | 3/3 | 6.44, 7.81, 6.26 |

No MuJoCo warnings were recorded in these rollouts. This is 18 smoke episodes,
not a statistically reliable success-rate estimate or a hardware payload claim.
The larger lift is not itself a demonstration of a better controller.

## Remaining validation before retraining/deployment

1. Obtain the calibrated reaching checkpoints and matching normalizers to replay
   the teammate's existing reaching acceptance result; they are not present in
   this checkout. Historical reports are not a newly reproduced evaluation.
2. Define and validate a manipulation-specific full servo model before claiming
   calibrated pick-and-lift or changing defaults for existing checkpoints.
3. Run a larger held-out placement/mass comparison and compare classical
   baselines before selecting a retraining experiment.
4. Validate on hardware at the calibration's P/D settings; simulation success
   does not approve autonomous physical operation.

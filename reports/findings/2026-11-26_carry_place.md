# First complete reach, grasp, carry and place sequence

**A complete scripted simulation demonstration now works:** reach, grasp, lift,
carry to a distinct destination, lower onto the table, open the gripper, retreat
and leave the cube at rest. The improved candidate passes **59/60 fresh complete
sequences**, with all 60 pickups retained and no sampled robot/table contacts or
MuJoCo warnings. It does **not** pass the frozen per-condition acceptance gate:
one 5 g / +30-degree case safely stops at a lowering timeout without releasing.

No RL training occurred. This is an opt-in complete-task prototype, not a change
to main, saved policies, old rewards/observations, model, gains or existing
pick/lift presets. Integration base commit: `f92d4e6`.

## Task and implementation

The existing validated `make_timed_env` performs its original **200-step pickup**
unchanged. After its normal Gym truncation, a separate backend-level controller
commands the same calibrated servo physics for the additional transfer phases.
It does not call `Gym.step` after truncation, extend the old training horizon or
represent itself as a ready-made Stable-Baselines3 environment.

Destination: rotate the initial cube position by 12 degrees toward the workspace
center, at the same radius around the pan axis. This moves the cube approximately
3–5 cm laterally while staying in the sampled radial workspace. The target cube
center is at the actual table resting height, 0.4875 m for this scene's 25 mm cube.
The destination is a coordinate, not a second physical object or edited MJCF site.
Saved images show a non-colliding magenta destination marker where visible.

New `control/carry_place.py` defines typed configuration, physical trace records
and a separate sequence controller:

1. Keep the established pickup and lift.
2. Carry high to the destination XY, retaining the closed jaw and requiring
   continued two-pad grasp throughout transport.
3. Lower with actual cube-position feedback. Require five stable samples of
   table support, XY error <=5 mm, resting-height error <=5 mm and translational
   speed <=0.01 m/s before opening.
4. Open slowly at 0.025 rad per period while holding the arm command. Loss of
   gripper contact is intentional only in this supported release phase.
5. Command a 50 mm upward retreat, requiring at least 40 mm measured upward
   clearance, then observe 60 further periods (1.98 seconds).
6. Complete success requires the cube supported on the table, not grasped,
   <=10 mm horizontal error, <=5 mm resting-height error and low final speed.

Transfer motion uses the existing position-first jaw-axis IK, a horizontal
grasp-axis preference, original gravity feedforward and 0.02 rad arm rate.
Cartesian references advance by at most 2 mm per control period and wait if
tracking lags more than 12 mm. Joint command segments and jaw opening are sampled
with the existing 4 mm robot/table clearance check. Unexpected grasp loss,
robot/table contact, warnings, rejected IK/clearance or phase timeout stops
the sequence rather than permitting an unsupported release.

Carry/lower/release/retreat/settle budgets are 150/150/60/100/60 periods: at most
520 additional backend periods after acquisition. The new demonstration has a
larger total task duration by design; it is not a 200-step full pick-and-place
claim. The original 200-step pick/lift task remains unchanged.

## Development and the preserved first result

Bring-up uses seed 87000, 10 g, nominal cube yaw. Resetting each tiny command from
the measured pose initially stalled under tracking error; an integrated Cartesian
reference corrects that while keeping the original physical parameters. The
first successful transfer/release was incorrectly rejected for missing the exact
50 mm retreat target despite about 43 mm actual retreat. An explicit >=40 mm
upward clearance criterion defines the intended retreat requirement.

The first frozen transfer trial uses seeds 88000–88004, four masses 5/10/15/20 g
and yaw offsets -30/0/+30 degrees: **60 cases**. It completes 34/60; all 60
acquisitions pass. The 26 failures are lowering timeouts, mostly 1–4 mm above
the table, with the cube still grasped. Several have intermittent contact but
not the required stable support window. No release was authorized in these
failed cases. The full first trial is retained under
`experiments/2026-11-26_carry_place/`; its failed gate was not overwritten.

## Bounded touchdown candidate

New `control/supported_place.py` subclasses the transfer controller and changes
only the lower-phase reference: permit at most **3 mm below the nominal resting
center** as a bounded support search. Actual stable support still ends lowering.
This is not a command to move the real cube through the table: physics contacts,
clearance checks and unchanged support/readiness gates remain active. No gains,
contact settings, collision masks or success tolerances were changed.

The recorded 5 g / nominal yaw / seed 88002 hover case was used for bring-up.
The candidate was then frozen before a full recorded-case recheck and new seeds.
The same 60 recorded cases improve from **34/60 to 60/60**, with zero new failures,
robot/table contacts or warnings. This is the matched comparison supporting the
touchdown change; it is not independent validation.

Fresh validation uses seeds **89000–89004**, the same four masses and three yaw
offsets, fixed nominal arm effects and original pickup. Results:

| Outcome | Result |
|---|---:|
| Successful acquisition | 60/60 |
| Complete carry/place/release/retreat | **59/60** |
| Supported, released rest within 10 mm among completed cases | 59/59 |
| Robot/table contact steps | 0 |
| MuJoCo warning steps | 0 |
| Final XY error among completed cases, mean / maximum | 5.338 / 6.892 mm |

All mass/yaw cells pass 5/5 except 5 g / +30 degrees, which passes 4/5. The frozen
gate requires >=95% in **each cell**, so `passed_gate` is **false** despite 98.3%
overall success. Only five independent placement centers are repeated across
masses and angles; do not treat 60 as independent placements or claim hardware
reliability. The candidate was not retuned after seeing this fresh result.

The remaining case is seed **89000**, 5 g, +30-degree yaw. Its destination is
(0.018996, -0.077785, 0.487500) m. After 150 lowering periods it remains grasped,
without table support, at about 3.36 mm above resting height and 5.25 mm XY error.
It stops with `lower_timeout`; the release guard correctly refuses to open.
There is no claim that a longer timeout alone would solve this tracking issue.

## Demonstration and remaining work

The supported seed-87000 demo completes in 461 periods total: pickup 200,
carry 31, lower 77, release 60, retreat 33 and settle 60, about **15.21 seconds**.
Destination is **(0.012411, 0.008813, 0.487500) m**; final cube center is
(0.017946, 0.007673, 0.487392) m, **5.65 mm horizontal error**. It is supported,
released and at rest after retreat, with zero sampled robot/table contacts or
warnings. Phase images and the complete physical qpos trace are saved.

The full pipeline is now implemented and demonstrated in simulation. Before
promoting it or defining complete-task training, fix and recheck the one known
last-centimeter placement case, preserving the support requirement rather than
loosening it. Then package a full-task environment with explicit destination,
termination and reward semantics if RL is required. This script intentionally
does not change the original RL environment to pretend it already trains placing.

These results use privileged cube pose/contact truth, the simulated gripper and
nominal calibrated arm effects. They do not validate physical placement force,
hardware orientation sensing or a neural policy's ability to perform the new task.
The current reaching and pickup policies remain separate and unchanged.

## Tests, files and reproduction

Tests were added first and failed before implementation. **438 tests pass,
one optional LeRobot test skips.** New tests cover reachable distinct destinations,
support/position/speed release conditions, an actual complete sequence after
retreat, refusal to open an unsupported held cube, and the bounded-probe repair
of a recorded light-cube hover. Full protected model/calibration/runtime/old
policy hashes match around both frozen trials and the recorded recheck.

New files:

- `src/rbe_501_so101/control/carry_place.py`
- `src/rbe_501_so101/control/supported_place.py`
- `scripts/sim2real/analyze/carry_place_check.py`
- `tests/test_carry_place.py`, `tests/test_supported_place.py`
- This finding and both new experiment directories.

Existing runtime files, model, old experiment results, training configuration,
checkpoint binaries and user-local CUDA dependency edits remain unchanged.
The runner keeps the initial sequence as its default and explicitly selects the
bounded-support candidate with `--supported`.

```powershell
# Watch the complete supported demonstration: run physics once, then play its recorded poses
uv run --locked python -m scripts.sim2real.analyze.carry_place_check --supported --replay --seed 87000 --mass-kg 0.01 --viewer --out experiments/place_viewer_1

# Repeat the frozen supported-candidate validation into a NEW directory
uv run --locked python -m scripts.sim2real.analyze.carry_place_check --supported --out experiments/supported_place_repeat
```

Output directories must be new. Omit `--viewer` for a headless replay with
phase PNGs and qpos trace. Viewer playback uses the recorded physics trajectory;
it does not run a second policy or learning update. The retained baseline run
configuration and local checkpoint companions are used for verification; the
runner does not load or train a neural policy.

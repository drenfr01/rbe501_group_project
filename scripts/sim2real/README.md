# Sim-to-real scripts (reach sub-task)

Spec: [`docs/sim2real_reach_transfer_spec.md`](../../docs/sim2real_reach_transfer_spec.md).
Findings go to [`reports/`](../../reports/README.md).

- `collect/` talks to the physical arm. Its output is irreplaceable raw data, so these scripts refuse to run on a
  dirty git tree unless `--allow-dirty` is given. Most take `--sim`, which runs the identical code path on MuJoCo
  through `SimJointBus` (robot id `sim`) as a dry run.
- `analyze/` is sim or offline only and can be rerun from saved data at any time.
- Every output folder gets a `manifest.json` (script, argv, commit, package versions, start/finish time), plus
  `uncommitted.patch` when the tree was dirty. A folder whose manifest has no `finished_at` is a crashed or aborted run.
- Scripts import shared code from `rbe_501_so101` only, never from each other.

Run from the repo root in the `lerobot` conda env
(`source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot`).

| Script | Phase | Needs arm | Writes | Example |
|---|---|---|---|---|
| `collect/jog.py` | tooling | yes | nothing | `python scripts/sim2real/collect/jog.py --port /dev/so101_follower` |
| `collect/sag_check.py record-poses` | 0 | yes (torque off, move by hand) | `data/sag_check/poses/poses.json` | `python scripts/sim2real/collect/sag_check.py record-poses` |
| `collect/sag_check.py run` | 0 | yes (`--sim` for a dry run) | `data/sag_check/<date>_<robot_id>/session.json`, manifest | `python scripts/sim2real/collect/sag_check.py run` |
| `collect/mirror_arm.py` | 1, and the Phase 3 dry run | yes (torque off) | `data/joint_mapping/<date>_<robot_id>/references.json`, manifest | `python scripts/sim2real/collect/mirror_arm.py` |
| `collect/run_policy_real.py` | 3 | yes (`--sim` for a dry run / prediction) | `experiments/<date>_real_reach/<controller>_<robot_id>_<time>/` (episodes, summary.json, README, manifest) | `python scripts/sim2real/collect/run_policy_real.py --policy ik_hold --targets data/targets/reach_repeat.json --repeats 10` |
| `analyze/analyze_sag.py` | 0 | no | `experiments/<date>_sag_analysis/` (report.json, README, real_sag.png), `data/calibration/actuator_calibration.json`; `--report-figures` copies `real_sag.png` to `reports/figures/` | `python scripts/sim2real/analyze/analyze_sag.py data/sag_check/<date>_rbe501_follower --report-figures` |
| `analyze/robustness_sweep.py` | 2 | no | `experiments/<date>_reach_robustness/` (report.json, README, kp_sweep.png, saturation.png); `--report-figures` copies the plots to `reports/figures/` | `python scripts/sim2real/analyze/robustness_sweep.py --calibration data/calibration/actuator_calibration.json --session data/sag_check/<date>_rbe501_follower --report-figures` |

## Order of a hardware campaign

1. `sag_check.py record-poses` (about 10 poses clear of the table and the joint limits), then `sag_check.py run`.
2. `mirror_arm.py`: confirm every joint sign and record physical references (link levels, stops) to fix the
   offsets in `HardwareConfig.joint_map`; re-measure at a second reference.
3. `analyze_sag.py` on the session (needs the Phase 1 mapping), which writes the calibration file.
4. `robustness_sweep.py --calibration ... --session ...` for the transfer-or-retrain decision.
5. `run_policy_real.py` for each of `ik_hold`, `gravity_comp`, `integral`, `trained` on
   `data/targets/reach_fixed.json`, then 10 repeats on `data/targets/reach_repeat.json`.
   Run each first with `--sim --calibration ...` to get the calibrated-sim prediction.

## Targets

`data/targets/reach_fixed.json` holds 10 targets from the training sampler, keeping only those whose IK pose is
inside the real arm's calibrated range minus `limit_margin_rad` (see `reports/differences.md`, D-02).
`reach_repeat.json` is the first 3 of them. Both are sim world coordinates in metres.

## Safety

The physical power switch is the e-stop and stays within reach during every powered session. `q` triggers the
software abort in `run_policy_real.py` (hold, then ramp to rest). Torque is only cut after ramping to rest.
Every commanded pose passes `SafetyMonitor` (table clearance, temperature, load, loop overruns).

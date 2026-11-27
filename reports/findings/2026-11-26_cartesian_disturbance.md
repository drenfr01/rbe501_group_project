# Frozen v3 policy: paired holding-disturbance evaluation

**Both controllers pass the tested disturbances; RL adds no demonstrated useful
robustness benefit.** Each retains 640/640 pushed holds and passes 40/40 no-push
controls. After subtracting normal drift, RL produces slightly more cube motion
and more pinch-frame motion. Keep the validated script as the reference; do not
launch longer training on the strength of this result.

## Results

| Outcome | Scripted | Frozen v3 SAC |
|---|---:|---:|
| Eligible pre-pulse holds | 680/680 | 680/680 |
| Final successful pushed holds | 640/640 | 640/640 |
| Successful no-force controls | 40/40 | 40/40 |
| Sampled transient or final loss of success | 0 | 0 |
| Sampled loss of two-pad grasp | 0 | 0 |
| Robot/table contact episodes | 0 | 0 |
| Initial approach contact episodes | 0 | 0 |
| MuJoCo warning episodes | 0 | 0 |

All four force levels pass in each direction and mass condition. No sampled
grasp losses or drops were observed. Every eligible trial confirms ten consecutive
successful samples in the first ten post-pulse steps (330 ms). Since none loses
task success, this establishes maintained holding, not a faster recovery from
a failed grasp. Contacts/grasp are sampled at 33 ms control boundaries, not
audited at every 1 ms physics step.

Among all 640 forced trials per controller, including any failures had they
occurred, descriptive means are:

| Physical motion metric | Scripted | SAC |
|---|---:|---:|
| Final-window cube RMS from pre-pulse pose | 0.4438 mm | 0.4343 mm |
| Final-window pinch RMS from pre-pulse pose | 0.0050 mm | 0.0875 mm |
| Peak cube displacement | 1.6352 mm | 1.6210 mm |
| Peak relative cube/pinch slip | 0.5992 mm | 0.5971 mm |

Maximum observed cube displacement is 4.532 mm scripted / 4.482 mm SAC. Raw cube
RMS is slightly lower for SAC, but its no-force drift also differs. The prespecified
force-minus-no-force comparison yields **+0.00842 mm RL-minus-script cube RMS**,
95% placement-cluster bootstrap interval **[+0.00262, +0.01501] mm**. Positive means
more disturbance-associated motion under RL. Adjusted pinch RMS difference is
**+0.05044 mm**, interval **[+0.03782, +0.06190] mm**. Neither is evidence of an RL
advantage. These are motion measures relative to each controller's own pre-pulse
pose, not end-effector target accuracy.

Retention difference is zero. Its empirical bootstrap interval [0, 0] reflects
identical success in this finite sample, **not a guarantee of population equivalence
or zero failure probability**. The useful-benefit gate fails; stability passes.
No controller or policy was retuned after seeing any result.

Evidence: `episodes.json` (all 1,360 outcomes), `paired_motion.json` (640 common
eligible comparisons), `assessment.json`, grouped `summary.json`, `retention.png`
and its plotting source, protocol, provenance and protected-input verification
under `experiments/2026-11-26_cartesian_disturbance/`. All protected model,
calibration, runtime source, old checkpoint, v3 policy/normalizer/configuration
checksums match before and after evaluation. Earlier results remain intact.

## Interpretation and next decision

The corrected script is already robust to this short-push ladder up to 0.8 N;
the test does not find the grip-failure boundary. These results do not justify
replacing it with the learned residual or treating more training transitions as
the next fix. Before longer training, choose an explicit task variation or
measurable objective where the script has a demonstrated weakness, and freeze
that benchmark and its reward criteria. Validate hardware jaw-angle/contact-force
mapping with the teammate who has the physical robot in parallel.

This is one frozen training seed, ten placement clusters, one contact/friction
model and fixed nominal arm effects. It tests neither randomized hardware/model
uncertainty nor real-robot behavior. It also does not isolate adaptive feedback
from changes in the learned controller's holding posture. A numerical improvement
on another earlier controller/test cannot be carried over to this policy.

## Protocol

This evaluates the saved Cartesian/0-degree-closing v3 SAC policy against its
zero-residual scripted reference. No training, controller, gain, model, reward,
normalization or task changes. Integration base commit: `b6e0469`.

The protocol was saved before rollouts under
`experiments/2026-11-26_cartesian_disturbance/protocol.json`:

- Ten new placement seeds 81000–81009, four masses 5/10/15/20 g.
- Training-domain calibrated stiffness plus fixed nominal servo/sensing effects.
  This sweep does not vary model parameters or randomize arm effects.
- No push, plus +/-world X and Y pushes at 0.1, 0.2, 0.4 and 0.8 N.
- Forces act at the cube COM without added torque, for three control periods:
  99 ms starting at 4.950 s. Each wrench is restored after the control period.
- Apply force only after ten consecutive successful task samples immediately
  before the pulse. The task success criterion remains a grasped cube lifted
  at least 4 cm. Ineligible acquisitions remain in overall outcomes.
- Original 200-step horizon: 1.551 s observation after force removal.
  Stable recovery means ten consecutive successful post-pulse samples; the
  earliest confirmation is 330 ms, not an estimate of physical settling time.
- Full design: 1,360 episodes, 680 per controller, comprising 40 unforced and
  640 forced cases per controller. Repeated mass/force/direction trials are
  clustered by ten placements, not treated as independent placements.

Primary comparison is paired final hold retention among common eligible
acquisitions, with placement-cluster bootstrap intervals (10,000 resamples,
seed 0). Motion metrics use true simulated cube/pinch poses rather than delayed
observations: final 30-step RMS relative to the pose immediately before the
pulse, peak cube displacement, and peak relative cube/pinch slip. Include failed
disturbed holds rather than censoring them. Subtract each controller's matching
no-force RMS before comparing RL minus scripted motion.

The frozen useful-benefit rule requires either a positive 95% retention-difference
interval or an adjusted cube-motion improvement interval entirely beyond 0.05 mm,
with adjusted pinch-motion increase no more than 0.05 mm. Neither route may add
paired final failures, robot/table contacts, initial approach contacts or warnings.
These practical thresholds were selected before this sweep; they were not used
to judge the earlier pilot retroactively. The assessment does not authorize full
training automatically.

## Validation and reproduction

New `scripts/sim2real/analyze/cartesian_disturbance.py` adds the fixed protocol,
typed outcome records, pairing, stable-window recovery and assessment. It reuses
the previously tested pulse application and physical-motion measurements.
Tests were added first and failed before implementation. **418 tests pass,
one optional LeRobot test skips.** New tests cover paired failures despite low
motion, ineligible acquisition denominators, consecutive recovery samples, and
exact physical-state equivalence of an unforced instrumented v3 rollout.
Existing tests check pulse boundaries, wrench restoration after exceptions,
physical impulse units and fixed-reference RMS.

```powershell
uv run --locked python -m scripts.sim2real.analyze.cartesian_disturbance --model-dir checkpoints/calibrated_cartesian_zero_v3_pilot --out experiments/cartesian_disturbance_repeat
```

Output directory must be new. The matching v3 model, observation statistics and
run configuration must exist locally. Checkpoints remain outside Git. Full-run
protected-input checks also require the older locally retained pilot artifacts.

# 2026-10-11 — SAC, centimeter-scale reward + smoothness penalty

**Result: did not pass.** Nominal mean holding error 14.8 mm (target < 5 mm),
p95 30.7 mm (target < 10 mm). Still worse than the IK-hold baseline (10.3 mm).

## What changed vs. the previous run

- Reward: `-100·d + exp(-(d/5 mm)²) - 0.01·‖a‖² - 0.01·‖a - a_prev‖²`
  (previously `-d + exp(...) - 0.01·‖a‖²`, no smoothness term).
- Target sampler rejects poses that collide with the table; grasp cube removed
  from the reach scene.
- Trained on the RTX 5070 Ti (three seeds in parallel, ~37 min).

Full settings are in `config.json`. Checkpoints: `checkpoints/reach_scaled/seed_{0,1,2}/`.

## Results (20 episodes per seed, last 30 of 150 steps)

| Policy | Mean (mm) | p95 (mm) | Per-seed mean (mm) | Hold speed (rad/s) |
|---|---|---|---|---|
| Previous reward, nominal | 15.0 | 40.7 | 14.2 / 9.4 / 21.4 | 0.087 |
| This run, nominal | 14.8 | 30.7 | 9.9 / 19.6 / 14.9 | 0.119 |
| This run, randomized dynamics | 14.8 | 30.4 | 9.8 / 19.5 / 15.1 | 0.134 |
| IK hold | 10.3 | 15.9 | — | 0.000 |
| Random | 345.1 | 535.3 | — | 0.408 |

![Training reward](training_reward.png)

![Holding error](holding_error.png)

## Takeaways

- Robustness improved a lot (randomized-dynamics p95 dropped from 142.6 mm to
  30.4 mm), but typical error did not move.
- Holding joint speed went up: the policy hovers near the target instead of
  settling, which fails the 0.05 rad/s stillness criterion by itself.
- Training reward was still slowly rising at 300k steps.

## Next experiments

1. Train to 1M steps to test whether more training alone closes the gap.
2. Inspect holding-window actions to confirm oscillation, then try a larger
   smoothness weight or a smaller per-step action limit near the target.

## Reproduce

```bash
python scripts/train_reach.py --algo sac --timesteps 300000 --seeds 0 1 2 --model-dir checkpoints/reach_scaled
python scripts/train_reach.py --eval --model-dir checkpoints/reach_scaled --num-episodes 20 > logs/eval_report_scaled.raw
python scripts/report_experiment.py --name 2026-10-11_sac_scaled_reward --model-dir checkpoints/reach_scaled \
    --eval-report logs/eval_report_scaled.raw --baseline-report logs/eval_report.json
```

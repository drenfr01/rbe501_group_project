# Calibrated-sim gate

**PASS**: every row within 1° (12/12 rows pass; stiffness-only sim: 3/12).

Errors are sim degrees along gravity (positive = sagged). Columns: real mean, sim mean (stiffness + stiction), mean |real - sim|, the same for the stiffness-only sim.

| source | joint | arrival | n | real | sim | diff | stiffness-only sim | its diff | pass |
|---|---|---|---|---|---|---|---|---|---|
| sag check, 8° approaches | shoulder_lift | with gravity | 5 | +0.23 | +0.34 | 0.21 | +1.69 | 1.46 | yes |
| sag check, 8° approaches | shoulder_lift | against gravity | 5 | +3.29 | +3.24 | 0.08 | +1.69 | 1.60 | yes |
| sag check, 8° approaches | elbow_flex | with gravity | 5 | +0.28 | +0.34 | 0.16 | +2.08 | 1.80 | yes |
| sag check, 8° approaches | elbow_flex | against gravity | 5 | +3.88 | +3.70 | 0.21 | +2.08 | 1.80 | yes |
| sag check, 3° approaches | shoulder_lift | with gravity | 12 | +1.35 | +1.45 | 0.47 | +2.81 | 1.47 | yes |
| sag check, 3° approaches | shoulder_lift | against gravity | 12 | +2.61 | +3.36 | 0.77 | +2.81 | 0.62 | yes |
| sag check, 3° approaches | elbow_flex | with gravity | 12 | +1.39 | +1.41 | 0.48 | +2.07 | 0.86 | yes |
| sag check, 3° approaches | elbow_flex | against gravity | 12 | +2.38 | +2.81 | 0.42 | +2.07 | 0.33 | yes |
| pose check, 30° moves | shoulder_lift | with gravity | 2 | +0.75 | +1.44 | 0.69 | +3.68 | 2.93 | yes |
| pose check, 30° moves | elbow_flex | against gravity | 2 | +4.17 | +4.15 | 0.08 | +2.30 | 1.87 | yes |
| pose check, 30° moves | shoulder_lift | against gravity | 1 | +4.98 | +5.27 | 0.29 | +3.16 | 1.82 | yes |
| pose check, 30° moves | elbow_flex | with gravity | 1 | +0.20 | +0.53 | 0.33 | +2.29 | 2.08 | yes |

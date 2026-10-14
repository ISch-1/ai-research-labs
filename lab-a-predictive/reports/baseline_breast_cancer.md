# Baseline — `breast_cancer`

Seed 42 · 3×10-fold CV (30 folds) · test size 20%

## Cross-validation (accuracy)

| Model | Mean ± std | vs `logreg` (corrected t-test p) |
|---|---|---|
| logreg | 0.9780 ± 0.0174 | — |
| random_forest | 0.9620 ± 0.0291 | 0.2474 |
| dummy | 0.6263 ± 0.0042 | 0.0000 |

## Held-out test set (95% bootstrap CI)

| Model | accuracy | balanced_accuracy | f1_macro |
|---|---|---|---|
| logreg | 0.9825 [0.9561, 1.0000] | 0.9812 [0.9504, 1.0000] | 0.9812 [0.9516, 1.0000] |
| random_forest | 0.9561 [0.9123, 0.9912] | 0.9504 [0.9031, 0.9878] | 0.9526 [0.9074, 0.9903] |
| dummy | 0.6316 [0.5439, 0.7193] | 0.5000 [0.5000, 0.5000] | 0.3871 [0.3523, 0.4184] |

McNemar `logreg` vs `random_forest` on the test set: p = 0.3750 (not significant at α = 0.05).

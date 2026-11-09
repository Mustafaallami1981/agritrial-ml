# Full case-study run vs earlier ad-hoc scripts

Run: `python -m agritrial_ml run configs/maize_spring2024.yaml --out results/maize_spring2024_run1`
(all configured permutations: 999 for LR/Ridge/SVR, 199 for RF; runtime 5405 s, single core).

| Output | Earlier script table | Result |
|---|---|---|
| performance.csv (LOTO) | T5 | identical (tested in tests/test_case_study_regression.py) |
| permutation_tests.csv | T6 | all 29 observed R2 and p-values identical |
| cv_comparison.csv | T7 (first 12 rows) | all 12 R2 identical (atol 1e-12) |
| subset_curve.csv | T8, pool E60 | 44/44 rows identical (max abs diff 0.0) |
| subset_curve.csv | T8, pool E30 | not configured in the package run (not compared) |
| selection_frequency.csv | T9 | different definition (k=3 from E60 vs top-3 from E30); not compared |

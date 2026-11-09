# Validator on the case-study workbook vs the earlier manual audit

The manual audit was done before the software existed (2026-11-08). The validator was then run on the same
sheet with the case-study configuration (evaluation/maize_validation_report.csv).

| # | Finding of the manual audit | Validator result |
|---|---|---|
| 1 | Grain yield (7N) = 500-grain weight (5N) x 0.05 in all plots | Detected: V07 warning, 100% of rows |
| 2 | Biological yield (10N) = 7N + 9N | Detected: V07 warning, 100% |
| 3 | Total chlorophyll (48N) = 46N + 47N | Detected: V07 info, 100% |
| 4 | Weed dry weight 60 d (17N) = 16N + 30 in weedy plots | Detected: V07 info, 80% |
| 5 | Uptakes = concentration x dry weight (31N, 32N, 33N, 39N, 44N exact; 38N partial) | Detected: V07 info (31N, 32N, 33N, 39N, 44N 100%; 38N 87%) |
| 6 | Identical replicates (40N, 27N, 15N-17N, 46N-48N, 50N, 52N-57N) | Detected: V08 (warning for 15N, used as predictor; info otherwise) |
| 7 | Replicates in exact arithmetic progression (25N, 26N, 34N-37N, 42N) | Detected: V08 (warning for 25N, 26N; info otherwise) |
| 8 | Harvest index (11N) = 7N / 10N x 100 | Not detected: ratios of two variables are not implemented |
| 9 | Protein (49N) ~ 40N x 6.25 (within 2% in 70/75 rows) | Not detected at the 0.5% tolerance (exact in 37% of rows) |
| 10 | Values differ between sheets of the workbook | Not detected: the tool reads one declared sheet |
| 11 | Treatment means of 16N differ from the thesis table | Not checked automatically: V10 needs a machine-readable reference table, which does not exist for the thesis |
| 12 | A second-season file equal to the first plus constants | Out of scope: the tool analyses one dataset per run |
| 13 | Unit labels inconsistent with values (e.g. stem diameter in 'm') | Not detected: semantic checks are not implemented |

Detected automatically: 7 of 13 finding types (1-7). The validator raised no warning that the manual audit had
not also found.

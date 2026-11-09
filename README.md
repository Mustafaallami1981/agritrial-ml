# agritrial-ml 0.1.0

Reproducible, leakage-aware machine-learning analysis of designed agricultural field trials.

The tool reads a trial workbook through a declarative configuration, validates the input, builds the
target, the treatment groups, the unit of analysis and timing-based predictor sets, evaluates models with
validation on treatments unseen in training, and exports every result with a run manifest (input and
output hashes, configuration, software versions, seed) and a log of excluded variables with reasons.

## Quick start

```
pip install -r requirements.txt
python -m agritrial_ml validate configs/maize_spring2024.yaml          # input checks only
python -m agritrial_ml run configs/maize_spring2024.yaml --out results/run1
python -m agritrial_ml run configs/maize_spring2024.yaml --out results/quick --n-perm 19   # fast run
python -m pytest -q                                                     # test suite
```

## Outputs of a run

| File | Content |
|---|---|
| `validation_report.csv` | every finding: check id, severity (error / warning / info), variable, message |
| `replicate_profile.csv` | per variable: share of identical replicates, of exact arithmetic progressions, within-treatment CV |
| `exclusions.csv` | variables excluded from all sets or from particular sets, with the reason |
| `analysis_table.csv` | the table actually modelled (treatment means or plots) |
| `anova_treatment_means.csv` | two-way ANOVA on treatment means (interaction as error) |
| `performance.csv`, `predictions.csv` | grouped (leave-one-treatment-out) R², RMSE, MAE and out-of-fold predictions |
| `permutation_tests.csv` | y-permutation and block-permutation tests |
| `cv_comparison.csv` | random row-level vs treatment-grouped validation |
| `subset_curve.csv`, `selection_frequency.csv` | in-fold top-k trait selection |
| `run_manifest.json` | configuration + hash, input file hash, software versions, seed, output hashes |

A run stops before modelling if validation reports an error; the validation report is still written.

## Layout

```
agritrial_ml/   config.py io.py validation.py design.py models.py evaluation.py runner.py __main__.py
configs/        maize_spring2024.yaml (case study)
data/           empty in this repository; see data/README.md (case-study workbook not redistributed)
tests/          unit, leakage-safety, reproducibility and regression tests; fixtures/ = SYNTHETIC test files
evaluation/     scripts and outputs that evaluate the software itself
docs/           requirements, architecture, inventory of the code before this project
```

The files in `tests/fixtures/` are synthetic software test cases with deliberately injected faults. They
are not agricultural data and are never used for any experimental result.

## Data

The case-study workbook (spring 2024 maize trial; Al-Bahadli et al., 2026) belongs to its authors and is
not part of this repository. Without it, the case-study regression test is skipped and all other tests run on
the synthetic fixtures. Holders of the file can check it against the SHA-256 in `data/README.md`.

## Citation

See `CITATION.cff`. Archived release and DOI on Zenodo: [DOI to be added after the first release].

## License

MIT (see `LICENSE`).

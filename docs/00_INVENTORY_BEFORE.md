# Inventory of the code that existed before the software project (2026-11-09)

Scope: the ad-hoc analysis scripts written for the spring-2024 maize reanalysis
(`maize_ml/*.py`, `scripts/load.py`, `scripts/a1.py`; 263 lines in total).

| Function | Status before | Where | Limitation |
|---|---|---|---|
| Read workbook sheet 1 into a table | Implemented | `scripts/load.py` | Column positions hard-coded for this workbook; 'zero' handled ad hoc in `a1.py` |
| Treatment means (unit of analysis) | Implemented | `common.py` | Hard-coded factor names (H, F); reads pickles from an absolute path |
| Predictor sets E30/E60, A, C | Implemented | `common.py`, `mlcore.py` | Lists typed by hand; timing not stored as metadata |
| Exclusions with reasons | Not implemented in code | comments + manuscript text | No machine-readable log |
| Input validation | Not implemented | exploratory scripts `scripts/a2*.py`–`a10.py` only | Not reusable, no report, no severity levels |
| Scaling inside folds | Implemented | `mlcore.py` (`make_pipeline`) | — |
| LOTO cross-validation | Implemented | `mlcore.loto` | Manual loop; no group-overlap check |
| Models (baseline, LR, Ridge, SVR, RF) | Implemented | `mlcore.MODELS` | Fixed settings, no registry/config |
| Permutation tests (y, block) | Implemented | `s3_perm.py` | Script-level, not reusable |
| Random vs grouped CV comparison | Implemented | `s4_leak_subset.py` | Script-level |
| In-fold top-k trait selection | Implemented | `s4_leak_subset.py` | Manual loop, not a pipeline step |
| Export of tables / figures | Partly | CSV per script, `s5_figs.py` | No run manifest, no data hash, no versions |
| Configuration file | Not implemented | — | — |
| Command-line interface | Not implemented | — | — |
| Automated tests | Not implemented | — | — |
| Reproducibility check | Not implemented | — | Results were re-run by hand only |

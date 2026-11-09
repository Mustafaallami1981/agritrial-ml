"""Generates SYNTHETIC SOFTWARE TEST FIXTURES with known, deliberately injected faults.

These files are NOT agricultural data and are never used for any experimental result. They exist only to
check that each validation rule fires (true positive) and that a clean file raises no error or warning.
Run: python tests/fixtures/make_fixtures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).parent
BANNER = "SYNTHETIC TEST FIXTURE - not agricultural data"


def clean_table(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for b in ["b1", "b2", "b3"]:
        for i, a in enumerate(["a0", "a1", "a2"]):
            for j, c in enumerate(["c0", "c1"]):
                y = 5 + 0.8 * i + 0.3 * j + rng.normal(0, 0.25)
                rows.append({"block": b, "A": a, "C": c,
                             "x_early": 10 + 2 * i + rng.normal(0, 1.5),
                             "x_mid": 3 + rng.normal(0, 0.6) + 0.2 * j,
                             "x_late": 50 + rng.normal(0, 5),
                             "yield": y})
    return pd.DataFrame(rows)


def base_config(csv_name):
    return {"data": {"file": csv_name, "first_row": 1, "tokens": {"zero": 0}},
            "design": {"block": 0, "factors": {"A": 1, "C": 2}, "unit": "treatment_mean"},
            "variables": [
                {"code": "x_early", "column": 3, "role": "predictor", "timing_dap_max": 20},
                {"code": "x_mid", "column": 4, "role": "predictor", "timing_dap_max": 40},
                {"code": "x_late", "column": 5, "role": "excluded", "reason": "measured at harvest"},
                {"code": "yield", "column": 6, "role": "target", "min": 0, "max": 20}],
            "predictor_sets": {"A": {"include_treatment": True, "include_traits": False, "max_timing_dap": 0},
                               "E": {"include_treatment": False, "max_timing_dap": 40},
                               "C": {"include_treatment": True, "max_timing_dap": 40}},
            "scopes": [{"name": "all"}],
            "models": {"Ridge": {"type": "ridge", "params": {"alpha": 1.0}},
                       "RF": {"type": "random_forest", "scale": False,
                              "params": {"n_estimators": 20, "random_state": 0, "max_features": "sqrt"}}},
            "evaluation": {"seed": 7,
                           "permutations": {"n": {"Ridge": 19, "RF": 9},
                                            "tests": [{"kind": "y", "scope": "all", "set": "E", "model": "Ridge"},
                                                      {"kind": "block", "scope": "all", "set": "C", "model": "Ridge"}]},
                           "cv_comparison": {"set": "E", "models": ["Ridge"], "n_splits": 3, "n_repeats": 2},
                           "subset_curve": {"pool_set": "E", "models": ["Ridge"], "report_k": 1}}}


def write(name, df, cfg_mut=None, raw_rows=None):
    csv = HERE / f"{name}.csv"
    out = df.copy()
    out.to_csv(csv, index=False)
    cfg = base_config(csv.name)
    if cfg_mut:
        cfg_mut(cfg)
    (HERE / f"{name}.yaml").write_text(f"# {BANNER}\n" + yaml.safe_dump(cfg, sort_keys=False))


# expected outcome per fixture: (check id, severity)
EXPECTED = {
    "f00_clean": None,
    "f01_missing_column": ("V01_columns", "error"),
    "f02_duplicate_key": ("V02_keys", "error"),
    "f03_unbalanced": ("V11_balance", "error"),
    "f04_non_numeric": ("V03_non_numeric", "error"),
    "f05_missing_target": ("V04_missing", "error"),
    "f06_out_of_range": ("V05_range", "warning"),
    "f07_predictor_no_timing": ("V06_timing", "error"),
    "f08_target_derived": ("V07_relation", "warning"),
    "f09_identical_replicates": ("V08_replicates", "warning"),
    "f10_arith_progression": ("V08_replicates", "warning"),
    "f11_reference_mismatch": ("V10_reference", "warning"),
    "f12_token_not_mapped": ("V03_non_numeric", "error"),
}


def main():
    d = clean_table()
    write("f00_clean", d)
    write("f01_missing_column", d, lambda c: c["variables"][0].update(column=12))
    write("f02_duplicate_key", pd.concat([d, d.iloc[[0]]], ignore_index=True))
    write("f03_unbalanced", d.drop(index=4).reset_index(drop=True))
    x = d.copy(); x["x_mid"] = x["x_mid"].astype(object); x.loc[2, "x_mid"] = "n.a."
    write("f04_non_numeric", x)
    x = d.copy(); x.loc[5, "yield"] = np.nan
    write("f05_missing_target", x)
    x = d.copy(); x.loc[7, "yield"] = -1.0
    write("f06_out_of_range", x)
    write("f07_predictor_no_timing", d, lambda c: c["variables"][1].pop("timing_dap_max"))
    x = d.copy(); x["x_early"] = x["yield"] / 0.05
    write("f08_target_derived", x)
    x = d.copy(); x["x_mid"] = x.groupby(["A", "C"])["x_mid"].transform("mean")
    write("f09_identical_replicates", x)
    x = d.copy(); m = x.groupby(["A", "C"])["x_mid"].transform("mean")
    off = x["block"].map({"b1": 0.0, "b2": 2.0, "b3": 1.0})
    x["x_mid"] = m - 1.0 + off
    write("f10_arith_progression", x)
    ref = d.groupby(["A", "C"])["yield"].mean().reset_index().rename(columns={"yield": "mean"})
    ref["variable"] = "yield"; ref.loc[0, "mean"] += 0.5
    ref.to_csv(HERE / "f11_reference.csv", index=False)
    write("f11_reference_mismatch", d, lambda c: c.setdefault("validation", {}).update(reference_means="f11_reference.csv"))
    x = d.copy(); x["x_early"] = x["x_early"].astype(object); x.loc[0, "x_early"] = "zero"
    write("f12_token_not_mapped", x, lambda c: c["data"].update(tokens={}))
    (HERE / "README.md").write_text(f"# {BANNER}\n\nFiles f00-f12 are generated by make_fixtures.py. Each injects one known fault "
                                    "(see EXPECTED in that script). They test the software only and are never mixed with the maize data or its results.\n")


if __name__ == "__main__":
    main()

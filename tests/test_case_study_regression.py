"""Regression test: the package reproduces the LOTO results of the earlier, independently written scripts
on the maize case-study data (skipped if the source workbook is not present)."""
import numpy as np
import pandas as pd
import pytest
from conftest import ROOT
from agritrial_ml.runner import run

CFG = ROOT / "configs" / "maize_spring2024.yaml"
DATA = ROOT / "data" / "maize_spring2024_source.xls"


@pytest.mark.skipif(not DATA.exists(), reason="case-study workbook not available")
def test_loto_matches_earlier_scripts(tmp_path):
    run(CFG, tmp_path, skip=("permutations", "subset", "cv_comparison", "anova"))
    new = pd.read_csv(tmp_path / "performance.csv")
    old = pd.read_csv(ROOT / "tests" / "reference" / "earlier_scripts_loto_performance.csv")
    old["model"] = old["model"].replace({"Baseline": "baseline"})
    m = old.merge(new, on=["scope", "set", "model"], suffixes=("_old", "_new"))
    assert len(m) == len(old) == 42
    ok = m[np.abs(m.R2_old) < 100]                # the underdetermined LR fit (R2 ~ -2173) is numerically unstable
    assert np.allclose(ok.R2_old, ok.R2_new, atol=1e-9)
    assert np.allclose(ok.RMSE_old, ok.RMSE_new, atol=1e-9)

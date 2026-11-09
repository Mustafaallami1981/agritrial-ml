import pandas as pd
import pytest
import yaml
from conftest import FIX
from agritrial_ml.config import ConfigError, load_config
from agritrial_ml.design import apply_scope, eligible_predictors, exclusion_log


def _tmp_cfg(tmp_path, mut):
    c = yaml.safe_load((FIX / "f00_clean.yaml").read_text())
    c["data"]["file"] = str(FIX / "f00_clean.csv")
    mut(c)
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(c)); return p


def test_excluded_needs_reason(tmp_path):
    p = _tmp_cfg(tmp_path, lambda c: c["variables"][2].pop("reason"))
    with pytest.raises(ConfigError):
        load_config(p)


def test_exactly_one_target(tmp_path):
    p = _tmp_cfg(tmp_path, lambda c: c["variables"][0].update(role="target"))
    with pytest.raises(ConfigError):
        load_config(p)


def test_timing_cutoffs_and_order():
    cfg = load_config(FIX / "f00_clean.yaml")
    assert eligible_predictors(cfg, 20) == ["x_early"]
    assert eligible_predictors(cfg, 40) == ["x_early", "x_mid"]
    assert eligible_predictors(cfg, 10) == []


def test_exclusion_log_has_reasons():
    log = exclusion_log(load_config(FIX / "f00_clean.yaml"))
    assert (log.reason.str.len() > 0).all()
    assert "x_late" in set(log.variable)


def test_scope_filter():
    t = pd.DataFrame({"A": ["a0", "a1", "a2"], "v": [1, 2, 3]})
    assert apply_scope(t, {"exclude": {"A": ["a1"]}}).A.tolist() == ["a0", "a2"]

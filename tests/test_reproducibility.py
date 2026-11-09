import json
from conftest import FIX
from agritrial_ml.runner import run


def test_two_runs_identical_outputs(tmp_path):
    m1 = run(FIX / "f00_clean.yaml", tmp_path / "r1")
    m2 = run(FIX / "f00_clean.yaml", tmp_path / "r2")
    assert m1["outputs_sha256"] == m2["outputs_sha256"]
    assert m1["data"]["sha256"] == m2["data"]["sha256"]
    man = json.loads((tmp_path / "r1" / "run_manifest.json").read_text())
    assert {"config_sha256", "software", "seed", "outputs_sha256"} <= set(man)


def test_validation_error_stops_run(tmp_path):
    import pytest
    from agritrial_ml.runner import ValidationFailed
    with pytest.raises(ValidationFailed):
        run(FIX / "f02_duplicate_key.yaml", tmp_path / "bad")
    assert (tmp_path / "bad" / "validation_report.csv").exists()
    assert not (tmp_path / "bad" / "performance.csv").exists()

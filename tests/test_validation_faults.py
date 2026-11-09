"""Fault-injection tests on SYNTHETIC fixtures (not agricultural data)."""
import importlib.util
import pytest
from conftest import FIX
from agritrial_ml.config import load_config
from agritrial_ml.runner import load_and_validate

spec = importlib.util.spec_from_file_location("mk", FIX / "make_fixtures.py")
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)


@pytest.mark.parametrize("name,expected", sorted(mk.EXPECTED.items()))
def test_each_fixture(name, expected):
    cfg = load_config(FIX / f"{name}.yaml")
    _, _, _, findings, _ = load_and_validate(cfg)
    found = {(f.check, f.severity) for f in findings}
    if expected is None:
        assert not [f for f in findings if f.severity in ("error", "warning")], findings
    else:
        assert expected in found, (name, found)

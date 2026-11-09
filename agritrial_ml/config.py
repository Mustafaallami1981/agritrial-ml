"""Configuration loading and schema checks (FR-1)."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from pathlib import Path
import yaml

ROLES = {"target", "predictor", "excluded"}


class ConfigError(ValueError):
    pass


@dataclass
class Variable:
    code: str
    column: object            # column index (int) or header name (str)
    name: str = ""
    unit: str = ""
    role: str = "excluded"
    timing_dap_max: float | None = None   # latest day after planting at which the value can exist
    stage: str = ""                        # free-text stage when no DAP is known
    reason: str = ""                       # required when role == excluded
    min: float | None = None
    max: float | None = None


@dataclass
class Config:
    raw: dict
    path: Path
    data: dict
    design: dict
    variables: list[Variable]
    predictor_sets: dict
    scopes: list
    models: dict
    evaluation: dict
    validation: dict = field(default_factory=dict)

    @property
    def target(self) -> Variable:
        return next(v for v in self.variables if v.role == "target")

    def var(self, code: str) -> Variable:
        for v in self.variables:
            if v.code == code:
                return v
        raise KeyError(code)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True, default=str).encode()).hexdigest()


def _require(d: dict, keys, where):
    missing = [k for k in keys if k not in d]
    if missing:
        raise ConfigError(f"{where}: missing key(s) {missing}")


def load_config(path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(raw, ["data", "design", "variables", "predictor_sets", "models", "evaluation"], "config")
    _require(raw["data"], ["file"], "data")
    _require(raw["design"], ["block", "factors"], "design")
    if not raw["design"]["factors"]:
        raise ConfigError("design.factors must list at least one factor")
    variables = []
    for i, v in enumerate(raw["variables"]):
        _require(v, ["code", "column"], f"variables[{i}]")
        var = Variable(**v)
        if var.role not in ROLES:
            raise ConfigError(f"variable {var.code}: role must be one of {sorted(ROLES)}")
        if var.role == "excluded" and not var.reason:
            raise ConfigError(f"variable {var.code}: an excluded variable needs a 'reason'")
        variables.append(var)
    codes = [v.code for v in variables]
    if len(codes) != len(set(codes)):
        raise ConfigError("variable codes must be unique")
    targets = [v for v in variables if v.role == "target"]
    if len(targets) != 1:
        raise ConfigError(f"exactly one target variable is required, found {len(targets)}")
    for name, s in raw["predictor_sets"].items():
        _require(s, ["include_treatment", "max_timing_dap"], f"predictor_sets.{name}")
    cfg = Config(raw=raw, path=path, data=raw["data"], design=raw["design"], variables=variables,
                 predictor_sets=raw["predictor_sets"], scopes=raw.get("scopes", [{"name": "all"}]),
                 models=raw["models"], evaluation=raw["evaluation"], validation=raw.get("validation", {}))
    return cfg


def resolve(cfg: Config, p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (cfg.path.parent / p).resolve()

"""Target, experimental groups, unit of analysis and timing-based predictor sets (FR-4)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import Config


def treatment_means(df: pd.DataFrame, X: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    factors = list(cfg.design["factors"])
    d = pd.concat([df[factors].astype(str).reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    return d.groupby(factors, sort=True).mean(numeric_only=True).reset_index()


def apply_scope(table: pd.DataFrame, scope: dict) -> pd.DataFrame:
    keep = np.ones(len(table), bool)
    for f, levels in (scope.get("exclude") or {}).items():
        keep &= ~table[f].astype(str).isin([str(l) for l in levels])
    return table.loc[keep].reset_index(drop=True)


def treatment_dummies(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    factors = list(cfg.design["factors"])
    return pd.get_dummies(table[factors].astype(str), drop_first=True).astype(float)


def eligible_predictors(cfg: Config, max_dap: float) -> list[str]:
    """Predictors whose latest possible measurement day is <= the cut-off.

    Column order is deterministic: variables that already qualify for an earlier (smaller) cut-off come
    first, then config order. Later sets therefore extend earlier ones by appending columns. (Order matters
    for random forests with max_features < 1, so it is fixed by rule rather than left implicit.)"""
    cutoffs = sorted({spec["max_timing_dap"] for spec in cfg.predictor_sets.values()})
    def tier(v):
        return next((c for c in cutoffs if v.timing_dap_max <= c), float("inf"))
    pool = [(tier(v), i, v.code) for i, v in enumerate(cfg.variables)
            if v.role == "predictor" and v.timing_dap_max is not None and v.timing_dap_max <= max_dap]
    return [c for _, _, c in sorted(pool)]


def exclusion_log(cfg: Config) -> pd.DataFrame:
    rows = []
    for v in cfg.variables:
        if v.role == "excluded":
            rows.append({"variable": v.code, "name": v.name, "decision": "excluded from all predictor sets", "reason": v.reason})
        elif v.role == "predictor":
            for s, spec in cfg.predictor_sets.items():
                if v.timing_dap_max is None or v.timing_dap_max > spec["max_timing_dap"]:
                    rows.append({"variable": v.code, "name": v.name, "decision": f"not in set {s}",
                                 "reason": f"timing_dap_max={v.timing_dap_max} > cut-off {spec['max_timing_dap']} DAP"})
    return pd.DataFrame(rows, columns=["variable", "name", "decision", "reason"])


@dataclass
class DesignData:
    table: pd.DataFrame          # one row per unit (treatment mean or plot)
    y: np.ndarray
    groups: np.ndarray           # treatment id per row
    sets: dict                   # set name -> DataFrame of predictors
    blocks: dict                 # set name -> (treatment block, trait block) for block permutation


def build(table: pd.DataFrame, cfg: Config) -> DesignData:
    factors = list(cfg.design["factors"])
    y = table[cfg.target.code].to_numpy(float)
    groups = table[factors].astype(str).agg("|".join, axis=1).to_numpy()
    A = treatment_dummies(table, cfg).reset_index(drop=True)
    sets, blocks = {}, {}
    for name, spec in cfg.predictor_sets.items():
        traits = eligible_predictors(cfg, spec["max_timing_dap"]) if spec.get("include_traits", True) else []
        T = table[traits].reset_index(drop=True)
        parts = ([A] if spec["include_treatment"] else []) + ([T] if traits else [])
        if not parts:
            continue
        sets[name] = pd.concat(parts, axis=1)
        if spec["include_treatment"] and traits:
            blocks[name] = (list(A.columns), traits)
    return DesignData(table=table, y=y, groups=groups, sets=sets, blocks=blocks)

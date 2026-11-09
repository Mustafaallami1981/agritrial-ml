"""Input validation (FR-3). Each check returns Findings; errors stop a run, warnings are logged."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import combinations
import numpy as np
import pandas as pd
from .config import Config

ERROR, WARNING, INFO = "error", "warning", "info"


@dataclass
class Finding:
    check: str
    severity: str
    variable: str
    message: str
    value: float | None = None

    def to_dict(self):
        return asdict(self)


def _treatment_id(df, factors):
    return df[factors].astype(str).agg("|".join, axis=1)


def check_columns(df, prov, cfg):
    return [Finding("V01_columns", ERROR, c, "configured column not found in the file") for c in prov.get("missing_columns", [])]


def check_keys(df, cfg):
    out = []
    factors = list(cfg.design["factors"])
    keys = ["block"] + factors
    if df[keys].isna().any().any():
        out.append(Finding("V02_keys", ERROR, ",".join(keys), "missing block or factor label in at least one row"))
        return out
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        out.append(Finding("V02_keys", ERROR, ",".join(keys), f"{int(dup.sum())} rows share the same block x treatment key", float(dup.sum())))
    # factorial balance: every factor-level combination in every block exactly once
    levels = [sorted(df[f].astype(str).unique()) for f in factors]
    expected = int(np.prod([len(l) for l in levels])) * df["block"].nunique()
    if len(df.drop_duplicates(keys)) != expected:
        out.append(Finding("V11_balance", ERROR, ",".join(keys),
                           f"design is not complete: {len(df.drop_duplicates(keys))} distinct block x treatment cells, {expected} expected"))
    return out


def check_numeric(bad_cells):
    return [Finding("V03_non_numeric", ERROR, c, f"row {r}: value {v!r} is not numeric and not in the token map") for r, c, v in bad_cells]


def check_missing(X, cfg):
    out = []
    for c in X.columns:
        n = int(X[c].isna().sum())
        if n:
            sev = ERROR if c == cfg.target.code else WARNING
            out.append(Finding("V04_missing", sev, c, f"{n} missing value(s)", float(n)))
    return out


def check_ranges(X, cfg):
    out = []
    for v in cfg.variables:
        if v.code not in X:
            continue
        s = X[v.code]
        if v.min is not None and (s < v.min).any():
            out.append(Finding("V05_range", WARNING, v.code, f"{int((s < v.min).sum())} value(s) below {v.min}", float(s.min())))
        if v.max is not None and (s > v.max).any():
            out.append(Finding("V05_range", WARNING, v.code, f"{int((s > v.max).sum())} value(s) above {v.max}", float(s.max())))
    return out


def check_timing(cfg):
    """Predictors must have a known timing; a predictor later than a set's cutoff is never used in that set."""
    out = []
    for v in cfg.variables:
        if v.role == "predictor" and v.timing_dap_max is None:
            out.append(Finding("V06_timing", ERROR, v.code, "predictor without timing_dap_max: cannot place it in an early-season set"))
    return out


def _share_mode(vals, decimals):
    r = np.round(vals[np.isfinite(vals)], decimals)
    if r.size == 0:
        return 0.0, None
    u, c = np.unique(r, return_counts=True)
    k = int(np.argmax(c))
    return c[k] / len(vals), float(u[k])


def check_relations(X, cfg, share=0.8, rel_tol=0.005):
    """Detect variables that are (almost) deterministic functions of others:
    ratio y = a*x, offset y = x + b, sum y = x1 + x2, product y = x1 * x2 (in >= `share` of rows).
    A relation involving the target is a leakage risk (warning); others are reported as info."""
    out = []
    num = X.select_dtypes(float)
    cols = [c for c in num.columns if num[c].std() > 0]
    A = {c: num[c].to_numpy() for c in cols}
    tcode = cfg.target.code
    def ordered(a, b):  # express the target (if involved) as the dependent variable
        return (b, a) if b == tcode else (a, b)
    for a_, b_ in combinations(cols, 2):
        y, x = ordered(a_, b_)
        yv, xv = A[y], A[x]
        tol = rel_tol * np.maximum(np.abs(yv), 1e-9)
        nz = (xv != 0) & (yv != 0)
        if nz.sum() >= 0.5 * len(yv):
            a = float(np.median(yv[nz] / xv[nz]))
            s = float(np.mean(np.abs(yv - a * xv) <= tol))
            if s >= share and abs(a - 1) > 1e-9:
                sev = WARNING if tcode in (x, y) else INFO
                out.append(Finding("V07_relation", sev, y, f"{y} = {a:.6g} x {x} in {s:.0%} of rows", a))
                continue
        _, b = _share_mode(yv - xv, 4)
        if b not in (0.0, None):
            s = float(np.mean(np.abs(yv - xv - b) <= np.maximum(tol, 1e-6)))
            if s >= share:
                sev = WARNING if tcode in (x, y) else INFO
                out.append(Finding("V07_relation", sev, y, f"{y} = {x} + {b:.6g} in {s:.0%} of rows", b))
    for y in cols:
        yv = A[y]
        scale = np.maximum(np.abs(yv), 1e-9)
        for x1, x2 in combinations([c for c in cols if c != y], 2):
            for op, val in (("+", A[x1] + A[x2]), ("x", A[x1] * A[x2])):
                ok = np.abs(yv - val) / scale <= rel_tol
                if ok.mean() >= share:
                    sev = WARNING if tcode in (y, x1, x2) else INFO
                    out.append(Finding("V07_relation", sev, y, f"{y} = {x1} {op} {x2} in {ok.mean():.0%} of rows", float(ok.mean())))
    return out


def replicate_profile(X, df, cfg):
    """Per variable: share of treatments with identical replicates, share with an exact arithmetic
    progression across replicates, and mean within-treatment CV (%)."""
    factors = list(cfg.design["factors"])
    tid = _treatment_id(df, factors)
    rows = []
    for c in X.columns:
        g = pd.DataFrame({"t": tid, "v": X[c]}).groupby("t")["v"]
        same = g.apply(lambda s: s.nunique() == 1).mean()
        def ap(s):
            a = np.sort(s.to_numpy())
            return len(a) >= 3 and len(np.unique(a)) == len(a) and np.allclose(np.diff(a), np.diff(a)[0], atol=1e-6)
        ap_share = g.apply(ap).mean()
        m = X[c].mean()
        cv = float(g.std().mean() / m * 100) if m else float("nan")
        rows.append({"variable": c, "identical_share": float(same), "arith_prog_share": float(ap_share), "within_cv_pct": cv})
    return pd.DataFrame(rows)


def check_replicates(prof, cfg):
    th = cfg.validation.get("replicates", {})
    t_same, t_ap, t_cv = th.get("identical_share", 0.5), th.get("arith_prog_share", 0.5), th.get("min_within_cv_pct", 0.5)
    out = []
    used = [v.code for v in cfg.variables if v.role in ("target", "predictor")]
    for _, r in prof.iterrows():
        flags = []
        if r.identical_share >= t_same:
            flags.append(f"identical replicates in {r.identical_share:.0%} of treatments")
        if r.arith_prog_share >= t_ap:
            flags.append(f"replicates form an exact arithmetic progression in {r.arith_prog_share:.0%} of treatments")
        if np.isfinite(r.within_cv_pct) and r.within_cv_pct < t_cv:
            flags.append(f"within-treatment CV {r.within_cv_pct:.2f}%")
        if flags:
            sev = WARNING if r.variable in used else INFO
            out.append(Finding("V08_replicates", sev, r.variable, "; ".join(flags)))
    return out


def check_reference(means: pd.DataFrame, ref: pd.DataFrame, cfg, rel_tol=0.005):
    """Compare computed treatment means with a reference table (e.g. a published table).
    `ref` columns: the design factors, 'variable', 'mean'."""
    out = []
    factors = list(cfg.design["factors"])
    for _, r in ref.iterrows():
        sel = np.logical_and.reduce([means[f].astype(str) == str(r[f]) for f in factors])
        if r["variable"] not in means or not sel.any():
            out.append(Finding("V10_reference", WARNING, str(r["variable"]), f"no computed mean for {dict(r[factors])}"))
            continue
        m = float(means.loc[sel, r["variable"]].iloc[0])
        if abs(m - r["mean"]) > rel_tol * max(abs(r["mean"]), 1e-9):
            out.append(Finding("V10_reference", WARNING, str(r["variable"]),
                               f"{dict(r[factors])}: computed {m:.4g} vs reference {r['mean']:.4g}", m - r["mean"]))
    return out


def validate(df, prov, X, bad_cells, cfg, reference: pd.DataFrame | None = None):
    findings = []
    findings += check_columns(df, prov, cfg)
    findings += check_keys(df, cfg)
    findings += check_numeric(bad_cells)
    findings += check_missing(X, cfg)
    findings += check_ranges(X, cfg)
    findings += check_timing(cfg)
    prof = None
    if not any(f.severity == ERROR and f.check in ("V01_columns", "V02_keys") for f in findings):
        findings += check_relations(X, cfg, **cfg.validation.get("relations", {}))
        prof = replicate_profile(X, df, cfg)
        findings += check_replicates(prof, cfg)
        if reference is not None:
            from .design import treatment_means
            findings += check_reference(treatment_means(df, X, cfg), reference, cfg)
    return findings, prof


def has_errors(findings):
    return any(f.severity == ERROR for f in findings)

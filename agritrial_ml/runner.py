"""End-to-end run: read -> validate -> design -> evaluate -> export (FR-7, FR-8)."""
from __future__ import annotations
import hashlib, json, platform, sys, time
from importlib import metadata
from pathlib import Path
import numpy as np
import pandas as pd
from . import __version__
from .config import Config, load_config, resolve
from .design import apply_scope, build, exclusion_log, treatment_means
from .evaluation import (block_permutation, grouped_predictions, metrics, random_predictions, y_permutation)
from .io import read_plot_table, to_numeric
from .models import make_model
from .validation import has_errors, validate


class ValidationFailed(RuntimeError):
    pass


def _versions():
    pk = {}
    for p in ("numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "PyYAML", "xlrd", "openpyxl"):
        try:
            pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pk[p] = None
    return {"agritrial_ml": __version__, "python": sys.version.split()[0], "platform": platform.platform(), **pk}


def load_and_validate(cfg: Config):
    df, prov = read_plot_table(cfg)
    codes = [v.code for v in cfg.variables]
    X, bad = to_numeric(df, codes, cfg.data.get("tokens", {}))
    X = X[codes]
    ref = None
    if cfg.validation.get("reference_means"):
        ref = pd.read_csv(resolve(cfg, cfg.validation["reference_means"]))
    findings, prof = validate(df, prov, X, bad, cfg, ref)
    return df, prov, X, findings, prof


def anova_means(means: pd.DataFrame, cfg: Config, codes):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    factors = list(cfg.design["factors"])
    safe = {f: f"fac{i}" for i, f in enumerate(factors)}   # factor names may clash with formula functions (e.g. 'C')
    rows = []
    for c in codes:
        d = means[factors + [c]].rename(columns={c: "y", **safe})
        m = smf.ols("y ~ " + " + ".join(f"C({safe[f]})" for f in factors), data=d).fit()
        a = sm.stats.anova_lm(m, typ=2)
        tot = a["sum_sq"].sum()
        r = {"variable": c}
        for f in factors:
            k = f"C({safe[f]})"
            r.update({f"F_{f}": a.loc[k, "F"], f"p_{f}": a.loc[k, "PR(>F)"], f"ss_share_{f}": a.loc[k, "sum_sq"] / tot})
        r["ss_share_residual"] = a.loc["Residual", "sum_sq"] / tot
        rows.append(r)
    return pd.DataFrame(rows)


def run(config_path, out_dir, n_perm_override=None, skip=()):
    t0 = time.time()
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, prov, X, findings, prof = load_and_validate(cfg)
    pd.DataFrame([f.to_dict() for f in findings], columns=["check", "severity", "variable", "message", "value"]).to_csv(out / "validation_report.csv", index=False)
    if prof is not None:
        prof.to_csv(out / "replicate_profile.csv", index=False)
    exclusion_log(cfg).to_csv(out / "exclusions.csv", index=False)
    if has_errors(findings):
        raise ValidationFailed(f"validation found errors; see {out/'validation_report.csv'}")

    ev = cfg.evaluation
    unit = cfg.design.get("unit", "treatment_mean")
    table = treatment_means(df, X, cfg) if unit == "treatment_mean" else pd.concat(
        [df[list(cfg.design["factors"])].astype(str), X], axis=1)
    table.to_csv(out / "analysis_table.csv", index=False)
    model_specs = cfg.models
    n_perm = dict(ev.get("permutations", {}).get("n", {}))
    if n_perm_override is not None:
        n_perm = {k: n_perm_override for k in model_specs}
    seed = int(ev.get("seed", 2026))

    if "anova" not in skip:
        pred_codes = [v.code for v in cfg.variables if v.role == "predictor"]
        anova_means(treatment_means(df, X, cfg), cfg, [cfg.target.code] + pred_codes).to_csv(out / "anova_treatment_means.csv", index=False)

    perf, preds, perms, comp, curve, freq, design_summary = [], [], [], [], [], [], {}
    for scope in cfg.scopes:
        tab = apply_scope(table, scope)
        D = build(tab, cfg)
        design_summary[scope["name"]] = {"units": int(len(tab)), "sets": {k: list(v.columns) for k, v in D.sets.items()}}
        base = make_model({"type": "baseline"})
        p = grouped_predictions(base, np.zeros((len(D.y), 1)), D.y, D.groups)
        perf.append({"scope": scope["name"], "set": "-", "model": "baseline", **metrics(D.y, p)})
        for sname, Xs in D.sets.items():
            for mname, spec in model_specs.items():
                m = make_model(spec)
                p = grouped_predictions(m, Xs.to_numpy(float), D.y, D.groups)
                rec = {"scope": scope["name"], "set": sname, "model": mname, **metrics(D.y, p),
                       "n_units": len(D.y), "n_predictors": Xs.shape[1],
                       "underdetermined": bool(Xs.shape[1] + 1 >= len(D.y) - 1 and spec["type"] == "linear")}
                perf.append(rec)
                preds.append(pd.DataFrame({"scope": scope["name"], "set": sname, "model": mname,
                                           "group": D.groups, "observed": D.y, "predicted": p}))
        # permutation tests
        if "permutations" not in skip:
            for t in ev.get("permutations", {}).get("tests", []):
                if t["scope"] != scope["name"] or t["set"] not in D.sets:
                    continue
                m = make_model(model_specs[t["model"]])
                n = n_perm.get(t["model"], 99)
                Xs = D.sets[t["set"]]
                if t["kind"] == "y":
                    obs, pval = y_permutation(m, Xs.to_numpy(float), D.y, D.groups, n, seed)
                else:
                    fixed, perm = D.blocks[t["set"]]
                    obs, pval = block_permutation(m, Xs, D.y, D.groups, fixed, perm, n, seed)
                perms.append({"scope": scope["name"], "test": t["kind"], "set": t["set"], "model": t["model"],
                              "R2_observed": obs, "n_permutations": n, "p_value": pval})
        # trait-subset curve (selection inside training folds)
        sc = ev.get("subset_curve")
        if sc and "subset" not in skip:
            pool = [c for c in D.sets[sc["pool_set"]].columns if c in [v.code for v in cfg.variables]]
            Xp = tab[pool].to_numpy(float)
            for k in range(1, len(pool) + 1):
                for mname in sc["models"]:
                    m = make_model(model_specs[mname], top_k=k)
                    p = grouped_predictions(m, Xp, D.y, D.groups)
                    curve.append({"scope": scope["name"], "pool": sc["pool_set"], "k": k, "model": mname, **metrics(D.y, p)})
            # selection frequency for the reported k
            from sklearn.model_selection import LeaveOneGroupOut
            from .models import TopKSpearman
            cnt = np.zeros(len(pool))
            for tr, _ in LeaveOneGroupOut().split(Xp, D.y, D.groups):
                cnt[TopKSpearman(sc.get("report_k", 3)).fit(Xp[tr], D.y[tr]).selected_] += 1
            n_f = len(np.unique(D.groups))
            freq += [{"scope": scope["name"], "pool": sc["pool_set"], "k": sc.get("report_k", 3), "variable": c,
                      "selected_share": cnt[j] / n_f} for j, c in enumerate(pool)]

    # validation-design comparison on plot rows
    cc = ev.get("cv_comparison")
    if cc and "cv_comparison" not in skip:
        plots = pd.concat([df[list(cfg.design["factors"])].astype(str), X], axis=1)
        for scope in cfg.scopes:
            tab = apply_scope(plots, scope)
            D = build(tab, cfg)
            Xs = D.sets[cc["set"]].to_numpy(float)
            for mname in cc["models"]:
                m = make_model(model_specs[mname])
                r2s = [metrics(D.y, random_predictions(m, Xs, D.y, cc.get("n_splits", 5), s))["R2"] for s in range(cc.get("n_repeats", 10))]
                comp.append({"scope": scope["name"], "model": mname, "design": f"random {cc.get('n_splits',5)}-fold, plot rows",
                             "n_rows": len(D.y), "R2": float(np.mean(r2s)), "R2_sd": float(np.std(r2s))})
                p = grouped_predictions(m, Xs, D.y, D.groups)
                comp.append({"scope": scope["name"], "model": mname, "design": "leave-one-treatment-out, plot rows",
                             "n_rows": len(D.y), "R2": metrics(D.y, p)["R2"], "R2_sd": float("nan")})

    pd.DataFrame(perf).to_csv(out / "performance.csv", index=False)
    if preds:
        pd.concat(preds).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(perms).to_csv(out / "permutation_tests.csv", index=False)
    pd.DataFrame(comp).to_csv(out / "cv_comparison.csv", index=False)
    pd.DataFrame(curve).to_csv(out / "subset_curve.csv", index=False)
    pd.DataFrame(freq).to_csv(out / "selection_frequency.csv", index=False)
    (out / "design_summary.json").write_text(json.dumps(design_summary, indent=2))

    outputs = {}
    for f in sorted(out.glob("*.csv")) + [out / "design_summary.json"]:
        outputs[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    manifest = {"config_file": str(Path(config_path).resolve()), "config_sha256": cfg.sha256(),
                "config": cfg.raw, "data": prov, "software": _versions(), "seed": seed,
                "permutations_used": n_perm, "skipped": list(skip),
                "validation_counts": {s: sum(f.severity == s for f in findings) for s in ("error", "warning", "info")},
                "runtime_seconds": round(time.time() - t0, 2), "outputs_sha256": outputs}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest

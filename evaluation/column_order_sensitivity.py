"""Software evaluation: sensitivity of random-forest LOTO R2 to the column order of the same predictors
(case-study treatment means). Motivates the deterministic column-order rule in design.eligible_predictors."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agritrial_ml.config import load_config
from agritrial_ml.design import apply_scope, build, treatment_means
from agritrial_ml.evaluation import grouped_predictions, metrics
from agritrial_ml.models import make_model
from agritrial_ml.runner import load_and_validate

cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "maize_spring2024.yaml")
df, prov, X, f, _ = load_and_validate(cfg)
tab0 = treatment_means(df, X, cfg)
rng = np.random.default_rng(11)
rows = []
for scope in cfg.scopes:
    D = build(apply_scope(tab0, scope), cfg)
    for s in ["E60", "C60"]:
        Xs = D.sets[s]
        r2 = []
        for i in range(20):
            perm = rng.permutation(Xs.shape[1]) if i else np.arange(Xs.shape[1])
            p = grouped_predictions(make_model(cfg.models["RF"]), Xs.iloc[:, perm].to_numpy(float), D.y, D.groups)
            r2.append(metrics(D.y, p)["R2"])
        rows.append({"scope": scope["name"], "set": s, "R2_rule_order": r2[0], "R2_min": min(r2), "R2_max": max(r2),
                     "R2_range": max(r2) - min(r2), "n_orders": len(r2)})
out = pd.DataFrame(rows); print(out.round(3).to_string())
out.to_csv(Path(__file__).parent / "column_order_sensitivity.csv", index=False)

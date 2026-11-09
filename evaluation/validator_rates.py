"""Software evaluation on SYNTHETIC data only: false-positive and detection rates of the heuristic
validation rules (V07 relations, V08 replicate structure) over many random clean trials.
Not agricultural data; results describe the validator, not any crop."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "fixtures"))
import make_fixtures as mk
from agritrial_ml.config import load_config
from agritrial_ml.runner import load_and_validate

N = 200
tmp = Path(__file__).parent / "_tmp"; tmp.mkdir(exist_ok=True)
res = {"clean_any_warning": 0, "clean_V07": 0, "clean_V08": 0,
       "inj_identical_detected": 0, "inj_ap_detected": 0, "inj_ratio_detected": 0, "inj_offset_detected": 0}

def findings_for(df, i):
    csv = tmp / f"t{i}.csv"; df.to_csv(csv, index=False)
    cfg = mk.base_config(str(csv)); p = tmp / f"t{i}.yaml"; p.write_text(yaml.safe_dump(cfg))
    return load_and_validate(load_config(p))[3]

for s in range(N):
    d = mk.clean_table(seed=1000 + s)
    f = findings_for(d, 0)
    res["clean_any_warning"] += any(x.severity in ("warning", "error") for x in f)
    res["clean_V07"] += any(x.check == "V07_relation" for x in f)
    res["clean_V08"] += any(x.check == "V08_replicates" for x in f)
    x = d.copy(); x["x_mid"] = x.groupby(["A", "C"])["x_mid"].transform("mean")
    res["inj_identical_detected"] += any(z.check == "V08_replicates" and z.variable == "x_mid" for z in findings_for(x, 1))
    x = d.copy(); m = x.groupby(["A", "C"])["x_mid"].transform("mean"); x["x_mid"] = m + x["block"].map({"b1": -.5, "b2": .5, "b3": 0.})
    res["inj_ap_detected"] += any(z.check == "V08_replicates" and z.variable == "x_mid" for z in findings_for(x, 2))
    x = d.copy(); x["x_early"] = x["yield"] * 20
    res["inj_ratio_detected"] += any(z.check == "V07_relation" and "x_early" in z.message for z in findings_for(x, 3))
    x = d.copy(); x["x_mid"] = x["x_early"] + 30; x.loc[x.index[:3], "x_mid"] += 1   # 15 of 18 rows exact
    res["inj_offset_detected"] += any(z.check == "V07_relation" and "x_mid" in z.message for z in findings_for(x, 4))
out = {k: v / N for k, v in res.items()}; out["n_datasets"] = N
print(json.dumps(out, indent=2))
(Path(__file__).parent / "validator_rates.json").write_text(json.dumps(out, indent=2))

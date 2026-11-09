"""Command-line interface: python -m agritrial_ml {validate,run} CONFIG [--out DIR]."""
import argparse, sys
import pandas as pd
from .config import load_config
from .runner import ValidationFailed, load_and_validate, run
from .validation import has_errors


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agritrial_ml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="read and validate the input only")
    v.add_argument("config")
    v.add_argument("--out", default=None, help="optional CSV path for the validation report")
    r = sub.add_parser("run", help="validate, model and export")
    r.add_argument("config")
    r.add_argument("--out", required=True)
    r.add_argument("--n-perm", type=int, default=None, help="override the number of permutations (e.g. for a quick run)")
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        cfg = load_config(a.config)
        _, _, _, findings, _ = load_and_validate(cfg)
        rep = pd.DataFrame([f.to_dict() for f in findings])
        if a.out:
            rep.to_csv(a.out, index=False)
        for s in ("error", "warning", "info"):
            print(f"{s}: {sum(f.severity == s for f in findings)}")
        return 1 if has_errors(findings) else 0
    try:
        m = run(a.config, a.out, a.n_perm)
    except ValidationFailed as e:
        print(e, file=sys.stderr)
        return 2
    print(f"done in {m['runtime_seconds']} s; outputs in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

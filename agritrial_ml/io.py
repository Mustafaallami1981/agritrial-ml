"""Reading trial workbooks into a tidy plot table (FR-2)."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd
from .config import Config, resolve


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_raw(path: Path, sheet) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path, header=None, dtype=object)
    engine = "xlrd" if suf == ".xls" else "openpyxl"
    return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, header=None, dtype=object, engine=engine)


def read_plot_table(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Return the plot table with raw cell values (no conversion) and provenance information.

    Columns: block, each design factor, and one column per configured variable code.
    Values are kept as read so that the validator can report non-numeric cells.
    """
    d = cfg.data
    path = resolve(cfg, d["file"])
    raw = _read_raw(path, d.get("sheet"))
    first, last = d.get("first_row", 0), d.get("last_row")
    body = raw.iloc[first: (last + 1 if last is not None else None)]
    cols = {"block": cfg.design["block"]}
    cols.update({f: c for f, c in cfg.design["factors"].items()})
    cols.update({v.code: v.column for v in cfg.variables})
    out = {}
    for name, col in cols.items():
        if isinstance(col, int):
            if col >= body.shape[1]:
                out[name] = pd.Series([pd.NA] * len(body), index=body.index, dtype=object)
                out[name].attrs["missing_column"] = True
                continue
            out[name] = body.iloc[:, col]
        else:
            header = raw.iloc[d.get("header_row", 0)]
            idx = [i for i, h in enumerate(header) if str(h).strip() == str(col)]
            if not idx:
                out[name] = pd.Series([pd.NA] * len(body), index=body.index, dtype=object)
                out[name].attrs["missing_column"] = True
                continue
            out[name] = body.iloc[:, idx[0]]
    df = pd.DataFrame(out).reset_index(drop=True)
    missing_cols = [n for n, s in out.items() if s.attrs.get("missing_column")]
    prov = {"file": str(path), "sha256": file_sha256(path), "sheet": d.get("sheet"),
            "rows_read": int(len(df)), "first_row": first, "last_row": last,
            "missing_columns": missing_cols}
    return df, prov


def to_numeric(df: pd.DataFrame, codes, tokens: dict) -> tuple[pd.DataFrame, list]:
    """Convert variable columns to float using an explicit token map (e.g. {'zero': 0}).

    Returns the converted frame and a list of (row, code, value) cells that could not be converted.
    Unconvertible cells become NaN and are reported, never silently filled.
    """
    out = df.copy()
    bad = []
    for c in codes:
        vals = []
        for i, x in enumerate(df[c].tolist()):
            if isinstance(x, str) and x.strip() in tokens:
                vals.append(float(tokens[x.strip()]))
                continue
            if x is None or (isinstance(x, float) and pd.isna(x)) or x is pd.NA or (isinstance(x, str) and not x.strip()):
                vals.append(float("nan"))
                continue
            try:
                vals.append(float(x))
            except (TypeError, ValueError):
                bad.append((i, c, x))
                vals.append(float("nan"))
        out[c] = vals
    return out, bad

"""TPC-H data for demos, tests and the dashboard (generated with DuckDB's `tpch` extension, cached as CSV)."""
from __future__ import annotations

import os
from typing import Any, Dict

TABLES = ["region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def data_dir(sf: float) -> str:
    """Where the CSVs for a scale factor live (matches the layout the C++ test harnesses use)."""
    plain = os.path.join(ROOT, "data", f"tpch_sf{sf}")
    if os.path.isdir(plain):
        return plain
    return os.path.join(ROOT, "data", f"tpch_sf{float(sf):g}")


def available_scale_factors() -> list:
    d = os.path.join(ROOT, "data")
    out = []
    if os.path.isdir(d):
        for name in os.listdir(d):
            if name.startswith("tpch_sf") and os.path.exists(os.path.join(d, name, "lineitem.csv")):
                try:
                    out.append(float(name[len("tpch_sf"):]))
                except ValueError:
                    pass
    return sorted(out)


def ensure_data(sf: float) -> str:
    """Generate (once) and return the CSV directory for `sf`."""
    ddir = data_dir(sf)
    if os.path.exists(os.path.join(ddir, "lineitem.csv")):
        return ddir
    import duckdb  # only needed for generation

    os.makedirs(ddir, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    for t in TABLES:
        con.execute(f"COPY {t} TO '{os.path.join(ddir, t + '.csv')}' (HEADER, DELIMITER ',')")
    return ddir


def load(db, sf: float, generate: bool = True) -> Dict[str, Dict[str, Any]]:
    ddir = ensure_data(sf) if generate else data_dir(sf)
    return {t: db.load_csv(t, os.path.join(ddir, t + ".csv")) for t in TABLES}

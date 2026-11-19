"""TPC-H benchmark: WarpDB CPU vs WarpDB GPU (cold + warm) vs DuckDB.  Writes a markdown table.

Usage: python tests/bench_tpch.py [--sf 1] [--runs 7] [--out docs/BENCHMARKS_TPCH.md]
"""
import argparse
import os
import statistics
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_test as dt  # noqa: E402


def median_ms(fn, runs):
    xs = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--exe", default=os.path.join(dt.ROOT, "build", "release", "warpdb.exe"))
    ap.add_argument("--out", default=os.path.join(dt.ROOT, "docs", "BENCHMARKS_TPCH.md"))
    args = ap.parse_args()

    ddir = os.path.join(dt.ROOT, "data", f"tpch_sf{args.sf}")
    duck = duckdb.connect()
    duck.execute("INSTALL tpch")
    duck.execute("LOAD tpch")
    duck.execute(f"CALL dbgen(sf={args.sf})")
    w = dt.Warp(args.exe)
    for t in dt.TABLES:
        w.run(f".load {t} {os.path.join(ddir, t + '.csv')}")

    # Process warm-up: the first GPU query pays one-time CUDA context + lazy kernel-module loading costs.
    # After this, "GPU cold" below means only "VRAM column cache is empty".
    w.run(".device gpu")
    for sql in dt.TPCH.values():
        w.query(sql.strip().rstrip(";"))

    rows = []
    for name, sql in dt.TPCH.items():
        sql1 = sql.strip().rstrip(";")
        d_ms = median_ms(lambda: duck.execute(sql1).fetchall(), args.runs)
        w.run(".device cpu")
        w.query(sql1)
        c_ms = median_ms(lambda: w.query(sql1), args.runs)
        w.run(".device gpu")
        w.run(".dropcache")
        t = time.perf_counter()
        w.query(sql1)  # cold: includes column upload over PCIe
        cold_ms = (time.perf_counter() - t) * 1000
        g_ms = median_ms(lambda: w.query(sql1), args.runs)
        rows.append((name, c_ms, cold_ms, g_ms, d_ms))
        print(f"{name}: cpu {c_ms:.1f}  gpu-cold {cold_ms:.1f}  gpu-warm {g_ms:.1f}  duckdb {d_ms:.1f}")
    w.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf8") as f:
        f.write(f"# TPC-H SF{args.sf:g} - WarpDB CPU vs GPU vs DuckDB\n\n")
        f.write(f"Median of {args.runs} runs, milliseconds, client-side wall time including result transfer. ")
        f.write("`GPU cold` starts with an empty VRAM column cache (includes the PCIe upload of the needed columns); ")
        f.write("`GPU warm` has the columns already resident.\n\n")
        f.write("| query | WarpDB CPU | WarpDB GPU cold | WarpDB GPU warm | DuckDB | GPU warm vs CPU | GPU warm vs DuckDB |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for n, c, cold, g, d in rows:
            f.write(f"| {n} | {c:.1f} | {cold:.1f} | {g:.1f} | {d:.1f} | {c / g:.1f}x | {d / g:.2f}x |\n")
        gm = lambda xs: statistics.geometric_mean(xs)  # noqa: E731
        f.write(f"| **geomean** | {gm([r[1] for r in rows]):.1f} | {gm([r[2] for r in rows]):.1f} | "
                f"{gm([r[3] for r in rows]):.1f} | {gm([r[4] for r in rows]):.1f} | "
                f"{gm([r[1] / r[3] for r in rows]):.1f}x | {gm([r[4] / r[3] for r in rows]):.2f}x |\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

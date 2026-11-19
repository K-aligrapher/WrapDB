"""Evaluates the cost-based CPU/GPU placement optimizer.

For every (scale factor, query, residency state) it measures
    cpu  : forced CPU execution
    gpu  : forced GPU execution            (cold state: VRAM column cache emptied before each run)
    auto : the optimizer's own choice      (cold state: VRAM column cache emptied before each run)
and compares auto with the ORACLE = faster of the two forced runs.

  hit    : auto picked the faster device (or was within 10% / 0.3 ms of the oracle - near ties don't count as misses)
  regret : (auto - oracle) / oracle

Usage: python tests/eval_placement.py [--sf 0.01 0.1 1] [--runs 5] [--out docs/PLACEMENT_EVAL.md]
"""
import argparse
import os
import statistics
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_test as dt  # noqa: E402

QUERIES = {**dt.TPCH}
for k in ["count_star", "min_max_date", "distinct", "having", "extract_group", "order_alias_limit", "join_alias",
          "three_way", "multi_key_join", "agg_in_arith", "like_mid", "between_dates", "or_and", "in_list_int",
          "count_expr", "select_star_limit", "str_range"]:
    QUERIES[k] = dt.CUSTOM[k]


def ensure_data(duck, sf):
    ddir = os.path.join(dt.ROOT, "data", f"tpch_sf{sf}")
    if not os.path.exists(os.path.join(ddir, "lineitem.csv")):
        os.makedirs(ddir, exist_ok=True)
        print(f"generating TPC-H sf={sf} ...", flush=True)
        duck.execute(f"CALL dbgen(sf={sf})")
        for t in dt.TABLES:
            duck.execute(f"COPY {t} TO '{os.path.join(ddir, t + '.csv')}' (HEADER, DELIMITER ',')")
    return ddir


def timed(w, sql, dev, cold, runs):
    """median wall-clock ms of `runs` executions on `dev`; returns (ms, device the engine actually used)."""
    w.run(f".device {dev}")
    if not cold:
        w.query(sql)  # warm: make sure the VRAM cache and the learned model have seen this query
    xs, used = [], ""
    for _ in range(runs):
        if cold:
            w.run(".dropcache")
        t = time.perf_counter()
        w.query(sql)
        xs.append((time.perf_counter() - t) * 1000)
        used = getattr(w, "last_device", "")
    return statistics.median(xs), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, nargs="+", default=[0.01, 0.1, 1])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--exe", default=os.path.join(dt.ROOT, "build", "release", "warpdb.exe"))
    ap.add_argument("--out", default=os.path.join(dt.ROOT, "docs", "PLACEMENT_EVAL.md"))
    args = ap.parse_args()

    duck = duckdb.connect()
    duck.execute("INSTALL tpch")
    duck.execute("LOAD tpch")
    records = []  # (sf, state, query, cpu, gpu, auto, chose)
    for sf in args.sf:
        ddir = ensure_data(duck, sf)
        runs = args.runs if sf <= 0.1 else max(3, args.runs - 2)
        w = dt.Warp(args.exe)
        for t in dt.TABLES:
            w.run(f".load {t} {os.path.join(ddir, t + '.csv')}")
        w.run(".device auto")
        w.query("select count(*) from region")  # triggers (or loads) calibration + GPU warm-up outside the timings
        for state in ("warm", "cold"):
            for name, sql in QUERIES.items():
                sql1 = sql.strip().rstrip(";")
                cpu, _ = timed(w, sql1, "cpu", state == "cold", runs)
                gpu, _ = timed(w, sql1, "gpu", state == "cold", runs)
                auto, chose = timed(w, sql1, "auto", state == "cold", runs)
                records.append((sf, state, name, cpu, gpu, auto, chose))
                oracle = min(cpu, gpu)
                print(f"sf{sf:<5g} {state:<4} {name:<18} cpu {cpu:8.2f}  gpu {gpu:8.2f}  auto {auto:8.2f} ({chose})  "
                      f"regret {max(0, auto - oracle) / oracle * 100:5.1f}%", flush=True)
        w.close()

    def hit(r):
        _, _, _, cpu, gpu, auto, chose = r
        oracle = min(cpu, gpu)
        best = "cpu" if cpu <= gpu else "gpu"
        return chose == best or auto <= 1.10 * oracle + 0.3

    def regret(r):
        _, _, _, cpu, gpu, auto, _ = r
        oracle = min(cpu, gpu)
        return max(0.0, auto - oracle) / oracle

    def high_stakes(r):
        """Decisions that matter: the two devices differ by >= 1.5x AND >= 1 ms."""
        _, _, _, cpu, gpu, _, _ = r
        return max(cpu, gpu) >= 1.5 * min(cpu, gpu) and max(cpu, gpu) - min(cpu, gpu) >= 1.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf8") as f:
        f.write("# Placement optimizer evaluation\n\n")
        f.write("Oracle = faster of forced-CPU and forced-GPU for the same query in the same state. "
                "*warm* = columns already resident in VRAM, *cold* = VRAM column cache emptied before every run "
                "(GPU pays the PCIe upload). All times are client-side wall-clock milliseconds, median of several runs.\n\n")
        f.write("Percentage regret is inflated for sub-millisecond queries (the optimizer's own ~0.3 ms is a large "
                "fraction of a 0.4 ms query). The *high-stakes* column counts only decisions where the two devices "
                "differ by at least 1.5x and 1 ms, and the totals show what a whole workload actually costs.\n\n")
        f.write("## Summary\n\n| scale | state | queries | hit rate | high-stakes hit rate | mean regret | auto total ms | always-CPU total | always-GPU total | oracle total |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        groups = sorted({(r[0], r[1]) for r in records})
        for g in groups + [("all", "all")]:
            rs = [r for r in records if g == ("all", "all") or (r[0], r[1]) == g]
            h = sum(hit(r) for r in rs) / len(rs)
            rg = [regret(r) for r in rs]
            hs = [r for r in rs if high_stakes(r)]
            hsr = f"{sum(hit(r) for r in hs) / len(hs) * 100:.0f}% ({len(hs)})" if hs else "n/a"
            f.write(f"| {g[0]} | {g[1]} | {len(rs)} | {h * 100:.0f}% | {hsr} | {statistics.mean(rg) * 100:.1f}% | "
                    f"{sum(r[5] for r in rs):.0f} | {sum(r[3] for r in rs):.0f} | {sum(r[4] for r in rs):.0f} | "
                    f"{sum(min(r[3], r[4]) for r in rs):.0f} |\n")
        f.write("\n## Per query\n\n| SF | state | query | CPU ms | GPU ms | auto ms | auto chose | oracle | regret |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in records:
            sf, state, name, cpu, gpu, auto, chose = r
            f.write(f"| {sf:g} | {state} | {name} | {cpu:.2f} | {gpu:.2f} | {auto:.2f} | {chose} | "
                    f"{'cpu' if cpu <= gpu else 'gpu'} | {regret(r) * 100:.0f}%{'' if hit(r) else ' ❌'} |\n")
    allr = records
    hs = [r for r in allr if high_stakes(r)]
    print(f"\nhit rate {sum(hit(r) for r in allr) / len(allr) * 100:.0f}%   high-stakes hit rate "
          f"{(sum(hit(r) for r in hs) / max(1, len(hs))) * 100:.0f}% ({len(hs)} decisions)   mean regret "
          f"{statistics.mean(regret(r) for r in allr) * 100:.1f}%   workload time auto/oracle "
          f"{sum(r[5] for r in allr) / sum(min(r[3], r[4]) for r in allr):.2f}x   wrote {args.out}")


if __name__ == "__main__":
    main()

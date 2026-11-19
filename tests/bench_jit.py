"""JIT pipeline fusion benchmark: interpreted (operator-at-a-time) GPU kernels vs NVRTC-fused kernels.

For each TPC-H query (columns resident in VRAM, medians of N runs):
  unfused ms : `.jit off`
  fused ms   : `.jit on`  (kernels already compiled)
  compile ms : NVRTC time of the very first run with an EMPTY kernel cache
  disk ms    : first run in a NEW process with the on-disk CUBIN cache populated (no NVRTC)
Usage: python tests/bench_jit.py [--sf 1] [--runs 9]
"""
import argparse
import os
import re
import shutil
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_test as dt  # noqa: E402


def jit_stats(w):
    text, _ = w.run(".jit stats")
    m = re.search(r"(\d+) compiles \(([\d.]+) ms in NVRTC\), (\d+) in-memory hits, (\d+) disk-cache hits \(([\d.]+) ms loading\)", text)
    return dict(compiles=int(m.group(1)), compile_ms=float(m.group(2)), mem=int(m.group(3)), disk=int(m.group(4)), load_ms=float(m.group(5)))


def timed(w, sql, runs):
    xs = []
    for _ in range(runs):
        t = time.perf_counter()
        w.query(sql)
        xs.append((time.perf_counter() - t) * 1000)
    return statistics.median(xs)


def open_session(args, ddir):
    w = dt.Warp(args.exe)
    for t in dt.TABLES:
        w.run(f".load {t} {os.path.join(ddir, t + '.csv')}")
    w.run(".device gpu")
    w.run(".jit minrows 0")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1)
    ap.add_argument("--runs", type=int, default=9)
    ap.add_argument("--exe", default=os.path.join(dt.ROOT, "build", "release", "warpdb.exe"))
    ap.add_argument("--out", default=os.path.join(dt.ROOT, "docs", "BENCHMARKS_JIT.md"))
    args = ap.parse_args()
    ddir = os.path.join(dt.ROOT, "data", f"tpch_sf{args.sf:g}")
    if not os.path.exists(os.path.join(ddir, "lineitem.csv")):
        ddir = os.path.join(dt.ROOT, "data", f"tpch_sf{args.sf}")
    cache = os.path.join(os.environ.get("LOCALAPPDATA", "."), "warpdb", "jit")
    shutil.rmtree(cache, ignore_errors=True)  # start from an empty kernel cache

    queries = dict(dt.TPCH)
    for k in ["count_star", "between_dates", "or_and", "expr_arith", "agg_in_arith", "extract_group", "count_expr"]:
        queries[k] = dt.CUSTOM[k]

    rows = []
    w = open_session(args, ddir)
    w.run(".jit off")
    for _ in range(40):  # ramp the GPU out of its idle clocks (interpreted kernels: compiles nothing we measure)
        w.query("select sum(l_quantity * l_extendedprice) from lineitem")
    for name, sql in queries.items():
        sql = sql.strip().rstrip(";")
        before = jit_stats(w)
        w.run(".jit on")
        t = time.perf_counter()
        w.query(sql)  # first run: uploads columns, compiles kernels
        first_ms = (time.perf_counter() - t) * 1000
        after = jit_stats(w)
        compile_ms = after["compile_ms"] - before["compile_ms"]
        # interleave the two variants so that GPU clock / thermal drift hits both equally
        for mode in ("off", "on", "off", "on"):  # warm-up of both code paths + clock ramp
            w.run(f".jit {mode}")
            w.query(sql)
        u, fz = [], []
        for _ in range(args.runs):
            for mode, bucket in (("off", u), ("on", fz)):
                w.run(f".jit {mode}")
                t = time.perf_counter()
                w.query(sql)
                bucket.append((time.perf_counter() - t) * 1000)
        unf, fused = statistics.median(u), statistics.median(fz)
        rows.append([name, unf, fused, compile_ms, None])
        print(f"{name:<16} unfused {unf:7.2f}  fused {fused:7.2f}  speedup {unf / fused:5.2f}x  NVRTC {compile_ms:7.1f} ms", flush=True)
    w.close()

    # new process, populated disk cache: kernels load from CUBIN files instead of being compiled
    w = open_session(args, ddir)
    for r in rows:
        sql = queries[r[0]].strip().rstrip(";")
        w.run(".jit off")
        w.query(sql)  # uploads the columns / loads interpreter kernels, so the next run isolates the JIT cost
        w.run(".jit on")
        before = jit_stats(w)
        t = time.perf_counter()
        w.query(sql)  # first fused run of this process: kernels come from the on-disk CUBIN cache
        r[4] = (time.perf_counter() - t) * 1000
        after = jit_stats(w)
        assert after["compiles"] == before["compiles"], "disk cache should avoid NVRTC"
    w.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    gm = lambda xs: statistics.geometric_mean(xs)  # noqa: E731
    with open(args.out, "w", encoding="utf8") as f:
        f.write(f"# JIT pipeline fusion - TPC-H SF{args.sf:g}\n\n")
        f.write("GPU, columns resident in VRAM, median of %d runs, client-side wall-clock ms. *Unfused* = operator-at-a-time "
                "kernels, every expression node materializes a vector; *fused* = NVRTC-compiled pipelines "
                "(filter+compact, and scan/filter/group/aggregate in one kernel). *NVRTC compile* is the one-off cost of "
                "the first run with an empty kernel cache; *new process* is the first fused run of a fresh process "
                "(columns already resident) that loads its kernels from the on-disk cache (no NVRTC).\n\n" % args.runs)
        f.write("| query | unfused ms | fused ms | speedup | NVRTC compile ms | first fused run, new process ms |\n|---|---|---|---|---|---|\n")
        for n, u, fz, c, d in rows:
            f.write(f"| {n} | {u:.2f} | {fz:.2f} | {u / fz:.2f}x | {c:.0f} | {d:.1f} |\n")
        f.write(f"| **geomean** | {gm([r[1] for r in rows]):.2f} | {gm([r[2] for r in rows]):.2f} | "
                f"{gm([r[1] / r[2] for r in rows]):.2f}x | {statistics.mean(r[3] for r in rows):.0f} (mean) | "
                f"{gm([r[4] for r in rows]):.1f} |\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

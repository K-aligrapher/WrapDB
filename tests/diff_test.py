"""Differential test: run SQL on WarpDB and on DuckDB, compare results.

Usage:  python tests/diff_test.py [--sf 0.1] [--exe build/release/warpdb.exe] [-k substring]
Data is generated with DuckDB's TPC-H extension into data/tpch_sf<sf>/ (git-ignored).
"""
import argparse
import csv
import datetime
import decimal
import io
import math
import os
import subprocess
import sys
import time

import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = ["region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]

# ---------------------------------------------------------------- queries
TPCH = {
    "q1": """select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice*(1-l_discount)) as sum_disc_price, sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order
        from lineitem where l_shipdate <= date '1998-12-01' - interval '90' day
        group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus""",
    "q3": """select l_orderkey, sum(l_extendedprice*(1-l_discount)) as revenue, o_orderdate, o_shippriority
        from customer, orders, lineitem where c_mktsegment = 'BUILDING' and c_custkey = o_custkey and l_orderkey = o_orderkey
        and o_orderdate < date '1995-03-15' and l_shipdate > date '1995-03-15'
        group by l_orderkey, o_orderdate, o_shippriority order by revenue desc, o_orderdate, l_orderkey limit 10""",
    "q5": """select n_name, sum(l_extendedprice*(1-l_discount)) as revenue
        from customer, orders, lineitem, supplier, nation, region
        where c_custkey = o_custkey and l_orderkey = o_orderkey and l_suppkey = s_suppkey and c_nationkey = s_nationkey
        and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'ASIA'
        and o_orderdate >= date '1994-01-01' and o_orderdate < date '1994-01-01' + interval '1' year
        group by n_name order by revenue desc""",
    "q6": """select sum(l_extendedprice*l_discount) as revenue from lineitem
        where l_shipdate >= date '1994-01-01' and l_shipdate < date '1994-01-01' + interval '1' year
        and l_discount between 0.06 - 0.01 and 0.06 + 0.01 and l_quantity < 24""",
    "q10": """select c_custkey, c_name, sum(l_extendedprice*(1-l_discount)) as revenue, c_acctbal, n_name, c_address, c_phone, c_comment
        from customer, orders, lineitem, nation
        where c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate >= date '1993-10-01'
        and o_orderdate < date '1993-10-01' + interval '3' month and l_returnflag = 'R' and c_nationkey = n_nationkey
        group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
        order by revenue desc, c_custkey limit 20""",
    "q12": """select l_shipmode,
        sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' then 1 else 0 end) as high_line_count,
        sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count
        from orders, lineitem where o_orderkey = l_orderkey and l_shipmode in ('MAIL', 'SHIP')
        and l_commitdate < l_receiptdate and l_shipdate < l_commitdate
        and l_receiptdate >= date '1994-01-01' and l_receiptdate < date '1994-01-01' + interval '1' year
        group by l_shipmode order by l_shipmode""",
    "q14": """select 100.00 * sum(case when p_type like 'PROMO%' then l_extendedprice*(1-l_discount) else 0 end)
        / sum(l_extendedprice*(1-l_discount)) as promo_revenue from lineitem, part
        where l_partkey = p_partkey and l_shipdate >= date '1995-09-01' and l_shipdate < date '1995-09-01' + interval '1' month""",
}

CUSTOM = {
    "count_star": "select count(*) from lineitem",
    "count_empty": "select count(*) as c from lineitem where l_quantity > 1000",
    "min_max_date": "select min(l_shipdate) as a, max(l_shipdate) as b, min(l_returnflag) as c, max(l_shipmode) as d from lineitem",
    "distinct": "select distinct l_returnflag, l_linestatus from lineitem order by l_returnflag, l_linestatus",
    "in_list_int": "select count(*) from orders where o_custkey in (1, 7, 22, 100, 4242)",
    "not_in_str": "select count(*) from lineitem where l_shipmode not in ('AIR', 'MAIL')",
    "like_mid": "select count(*) from part where p_name like '%green%'",
    "not_like": "select count(*) from part where p_type not like '%BRASS'",
    "having": "select l_suppkey, count(*) as c, sum(l_quantity) as q from lineitem group by l_suppkey having count(*) > 60 order by c desc, l_suppkey limit 25",
    "extract_group": "select extract(year from o_orderdate) as y, count(*) as c, avg(o_totalprice) as p from orders group by y order by y",
    "order_alias_limit": "select o_orderkey, o_totalprice as tp from orders order by tp desc, o_orderkey limit 15",
    "order_ordinal": "select o_orderpriority, count(*) from orders group by 1 order by 1",
    "expr_arith": "select sum(l_extendedprice * (1 - l_discount) / 2 + l_tax * 3 - 1) as x, sum(l_quantity % 7) as m from lineitem",
    "between_dates": "select count(*), sum(l_quantity) from lineitem where l_shipdate between date '1995-01-01' and date '1995-12-31'",
    "or_and": "select count(*) from lineitem where (l_quantity < 5 or l_quantity > 45) and not (l_discount = 0.05)",
    "str_range": "select count(*) from customer where c_mktsegment > 'BUILDING' and c_mktsegment <= 'MACHINERY'",
    "join_on_syntax": "select n_name, count(*) as c from supplier join nation on s_nationkey = n_nationkey group by n_name order by n_name",
    "join_alias": "select n.n_name, count(*) as c from supplier s join nation n on s.s_nationkey = n.n_nationkey where s.s_acctbal > 0 group by n.n_name order by c desc, n.n_name",
    "three_way": "select r_name, count(*) as c, sum(s_acctbal) as b from supplier, nation, region where s_nationkey = n_nationkey and n_regionkey = r_regionkey group by r_name order by r_name",
    "limit_only": "select count(*) from (select 1) t" ,  # unsupported on purpose -> expect error from both? skipped below
    "select_star_limit": "select * from region order by r_regionkey limit 3",
    "neg_literals": "select count(*) from lineitem where l_quantity > -5 and l_discount >= -0.5",
    "count_expr": "select count(l_orderkey) as a, count(*) as b from lineitem where l_returnflag = 'N'",
    "multi_key_join": "select count(*) from lineitem, partsupp where l_partkey = ps_partkey and l_suppkey = ps_suppkey",
    "agg_in_arith": "select l_returnflag, sum(l_quantity) / count(*) as avgq, max(l_extendedprice) - min(l_extendedprice) as spread from lineitem group by l_returnflag order by l_returnflag",
}
SKIP = {"limit_only"}


# ---------------------------------------------------------------- warpdb driver
class Warp:
    def __init__(self, exe):
        self.p = subprocess.Popen([exe, "--csv", "--protocol"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)

    def run(self, line):
        self.p.stdin.write(line.replace("\n", " ") + ("\n" if line.startswith(".") else " \n"))
        self.p.stdin.flush()
        out = []
        while True:
            l = self.p.stdout.readline()
            if not l:
                raise RuntimeError("warpdb exited unexpectedly:\n" + "".join(out))
            if l.startswith("@@END"):
                parts = l.split()
                self.last_device = parts[2] if len(parts) > 2 else ""
                return "".join(out), float(parts[1])
            if l.startswith("@@ERR"):
                raise RuntimeError(l[6:].strip())
            out.append(l)

    def query(self, sql):
        text, ms = self.run(sql.strip().rstrip(";") + ";")
        rows = list(csv.reader(io.StringIO(text)))
        return rows[0], rows[1:], ms

    def close(self):
        try:
            self.p.stdin.write(".quit\n")
            self.p.stdin.flush()
        except Exception:
            pass
        self.p.wait(timeout=10)


# ---------------------------------------------------------------- comparison
def norm_duck(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()[:10]
    return v


def same(a, b):
    if isinstance(b, float) or isinstance(b, int) and not isinstance(b, bool):
        try:
            x, y = float(a), float(b)
        except ValueError:
            return False
        return abs(x - y) <= 1e-6 * max(1.0, abs(x), abs(y))
    return str(a) == str(b)


def sort_key(row):
    return tuple((0, round(float(c), 4)) if isinstance(c, (int, float)) else (1, str(c)) for c in row)


def compare(wrows, drows, ordered):
    if len(wrows) != len(drows):
        return f"row count differs: warpdb={len(wrows)} duckdb={len(drows)}"
    drows = [[norm_duck(c) for c in r] for r in drows]
    wc = []
    for r in wrows:
        wc.append([float(c) if i < len(drows[0]) and isinstance(drows[0][i], (int, float)) and not isinstance(drows[0][i], bool) and c != "" else c
                   for i, c in enumerate(r)]) if drows else None
    if not ordered:
        wc.sort(key=sort_key)
        drows.sort(key=sort_key)
    for i, (w, d) in enumerate(zip(wc, drows)):
        if len(w) != len(d):
            return f"column count differs at row {i}: {len(w)} vs {len(d)}"
        for j, (a, b) in enumerate(zip(w, d)):
            if not same(a, b):
                return f"row {i} col {j}: warpdb={a!r} duckdb={b!r}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=0.1)
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "release", "warpdb.exe"))
    ap.add_argument("-k", default="", help="only run queries whose name contains this")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "auto"], help="WarpDB execution device")
    ap.add_argument("--jit", default="on", choices=["on", "off"], help="GPU JIT pipeline fusion (on = compile for every size)")
    args = ap.parse_args()

    ddir = os.path.join(ROOT, "data", f"tpch_sf{args.sf}")
    duck = duckdb.connect()
    duck.execute("INSTALL tpch")
    duck.execute("LOAD tpch")
    if not os.path.exists(os.path.join(ddir, "lineitem.csv")):
        os.makedirs(ddir, exist_ok=True)
        print(f"generating TPC-H sf={args.sf} ...")
        duck.execute(f"CALL dbgen(sf={args.sf})")
        for t in TABLES:
            duck.execute(f"COPY {t} TO '{os.path.join(ddir, t + '.csv')}' (HEADER, DELIMITER ',')")
    else:
        duck.execute(f"CALL dbgen(sf={args.sf})")

    w = Warp(args.exe)
    t0 = time.time()
    for t in TABLES:
        text, _ = w.run(f".load {t} {os.path.join(ddir, t + '.csv')}")
        print("  " + text.strip())
    print(f"loaded in {time.time() - t0:.1f}s\n")
    if args.device in ("gpu", "auto"):
        w.run(f".device {args.device}")
        if args.jit == "on":
            w.run(".jit minrows 0")  # exercise the fused kernels on every input size
        else:
            w.run(".jit off")
        print(f"device = {args.device} (each query runs twice; the second, warm run is timed)\n")

    failed = 0
    ran = 0
    print(f"{'query':<22}{'result':<8}{'rows':>8}{'warpdb ms':>12}{'duckdb ms':>12}")
    for name, sql in {**TPCH, **CUSTOM}.items():
        if name in SKIP or args.k not in name:
            continue
        ran += 1
        ordered = "order by" in sql.lower()
        try:
            t = time.time()
            duck_rows = duck.execute(sql).fetchall()
            dms = (time.time() - t) * 1000
            if args.device in ("gpu", "auto"):
                w.query(sql)  # first run: uploads the needed columns / lets the optimizer see the query
            _, wrows, wms = w.query(sql)
            err = compare(wrows, duck_rows, ordered)
        except Exception as e:  # noqa: BLE001
            err = f"exception: {e}"
            wms = dms = float("nan")
            duck_rows = []
        status = "ok" if err is None else "FAIL"
        print(f"{name:<22}{status:<8}{len(duck_rows):>8}{wms:>12.1f}{dms:>12.1f}")
        if err:
            failed += 1
            print(f"    -> {err}")
    w.close()
    print(f"\n{ran - failed}/{ran} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""The TPC-H subset WarpDB supports, plus a few illustrative extras (used by the dashboard and the benchmarks)."""

TPCH = {
    "Q1 pricing summary": """select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice*(1-l_discount)) as sum_disc_price, sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order
        from lineitem where l_shipdate <= date '1998-12-01' - interval '90' day
        group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus""",
    "Q3 shipping priority": """select l_orderkey, sum(l_extendedprice*(1-l_discount)) as revenue, o_orderdate, o_shippriority
        from customer, orders, lineitem where c_mktsegment = 'BUILDING' and c_custkey = o_custkey and l_orderkey = o_orderkey
        and o_orderdate < date '1995-03-15' and l_shipdate > date '1995-03-15'
        group by l_orderkey, o_orderdate, o_shippriority order by revenue desc, o_orderdate, l_orderkey limit 10""",
    "Q5 local supplier volume": """select n_name, sum(l_extendedprice*(1-l_discount)) as revenue
        from customer, orders, lineitem, supplier, nation, region
        where c_custkey = o_custkey and l_orderkey = o_orderkey and l_suppkey = s_suppkey and c_nationkey = s_nationkey
        and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'ASIA'
        and o_orderdate >= date '1994-01-01' and o_orderdate < date '1994-01-01' + interval '1' year
        group by n_name order by revenue desc""",
    "Q6 forecasting revenue": """select sum(l_extendedprice*l_discount) as revenue from lineitem
        where l_shipdate >= date '1994-01-01' and l_shipdate < date '1994-01-01' + interval '1' year
        and l_discount between 0.06 - 0.01 and 0.06 + 0.01 and l_quantity < 24""",
    "Q10 returned items": """select c_custkey, c_name, sum(l_extendedprice*(1-l_discount)) as revenue, c_acctbal, n_name, c_address, c_phone, c_comment
        from customer, orders, lineitem, nation
        where c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate >= date '1993-10-01'
        and o_orderdate < date '1993-10-01' + interval '3' month and l_returnflag = 'R' and c_nationkey = n_nationkey
        group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
        order by revenue desc, c_custkey limit 20""",
    "Q12 shipping modes": """select l_shipmode,
        sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' then 1 else 0 end) as high_line_count,
        sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count
        from orders, lineitem where o_orderkey = l_orderkey and l_shipmode in ('MAIL', 'SHIP')
        and l_commitdate < l_receiptdate and l_shipdate < l_commitdate
        and l_receiptdate >= date '1994-01-01' and l_receiptdate < date '1994-01-01' + interval '1' year
        group by l_shipmode order by l_shipmode""",
    "Q14 promotion effect": """select 100.00 * sum(case when p_type like 'PROMO%' then l_extendedprice*(1-l_discount) else 0 end)
        / sum(l_extendedprice*(1-l_discount)) as promo_revenue from lineitem, part
        where l_partkey = p_partkey and l_shipdate >= date '1995-09-01' and l_shipdate < date '1995-09-01' + interval '1' month""",
}

EXTRA = {
    "orders per year": "select extract(year from o_orderdate) as y, count(*) as orders, avg(o_totalprice) as avg_price "
                       "from orders group by y order by y",
    "top suppliers": "select l_suppkey, count(*) as lines, sum(l_quantity) as qty from lineitem group by l_suppkey "
                     "having count(*) > 500 order by lines desc, l_suppkey limit 25",
    "tiny dimension join": "select n_name, count(*) as suppliers from supplier join nation on s_nationkey = n_nationkey "
                           "group by n_name order by n_name",
    "count rows": "select count(*) from lineitem",
}

ALL = {**TPCH, **EXTRA}


def one_line(sql: str) -> str:
    return " ".join(sql.split())

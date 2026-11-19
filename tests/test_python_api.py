"""Tests for the `warpdb` Python package (pybind11 bindings + pandas API).

Run:  .\\.venv\\Scripts\\python -m pytest tests/test_python_api.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

warpdb = pytest.importorskip("warpdb", reason="build the extension first: .\\build.ps1")


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(7)
    n = 20000
    return pd.DataFrame({
        "id": np.arange(n),
        "g": rng.integers(0, 12, n),
        "x": np.round(rng.random(n) * 100, 3),
        "day": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 800, n), unit="D"),
        "tag": rng.choice(["alpha", "beta", "gamma", "delta"], n),
    })


@pytest.fixture(scope="module")
def db(df):
    d = warpdb.connect()
    d.set_jit(True, min_rows=0)
    d.load_dataframe("t", df)
    return d


def test_gpu_visible_and_tables_loaded(db):
    assert db.has_gpu and "RTX" in db.gpu_name or db.has_gpu
    t = {x["name"]: x for x in db.tables()}
    assert t["t"]["rows"] == 20000
    assert [c["name"] for c in t["t"]["columns"]] == ["id", "g", "x", "day", "tag"]


def test_groupby_matches_pandas_on_every_device(db, df):
    want = df.groupby("g").agg(n=("id", "count"), sx=("x", "sum"), mx=("x", "max")).reset_index()
    for dev in ("cpu", "gpu", "auto"):
        res = db.sql("select g, count(*) as n, sum(x) as sx, max(x) as mx from t group by g order by g", device=dev)
        got = res.to_pandas()
        assert res.device == ("cpu" if dev == "cpu" else res.device)
        assert list(got.columns) == ["g", "n", "sx", "mx"]
        np.testing.assert_array_equal(got["g"].to_numpy(), want["g"].to_numpy())
        np.testing.assert_array_equal(got["n"].to_numpy(), want["n"].to_numpy())
        np.testing.assert_allclose(got["sx"].to_numpy(), want["sx"].to_numpy(), rtol=1e-9)
        np.testing.assert_allclose(got["mx"].to_numpy(), want["mx"].to_numpy())


def test_column_dtypes_roundtrip(db, df):
    got = db.sql("select id, x, day, tag from t where id < 5 order by id", device="cpu").to_pandas()
    assert got["id"].dtype == np.int64
    assert got["x"].dtype == np.float64
    assert np.issubdtype(got["day"].dtype, np.datetime64)
    assert got["tag"].dtype == object
    pd.testing.assert_series_equal(got["day"].reset_index(drop=True), df["day"].iloc[:5].reset_index(drop=True),
                                   check_names=False, check_dtype=False)
    assert list(got["tag"]) == list(df["tag"].iloc[:5])


def test_string_filters_and_dates(db, df):
    q = "select count(*) as c from t where tag like 'a%' and day >= date '2021-01-01'"
    want = int(((df["tag"].str.startswith("a")) & (df["day"] >= "2021-01-01")).sum())
    for dev in ("cpu", "gpu"):
        assert int(db.sql(q, device=dev).to_pandas()["c"][0]) == want


def test_result_metadata_and_stages(db):
    res = db.sql("select g, sum(x) from t where x > 50 group by g", device="gpu")
    assert res.device == "gpu" and res.ms > 0 and res.nrows > 0
    stages = res.stages
    assert {"op", "device", "rows_in", "rows_out", "ms"} <= set(stages.columns)
    assert "aggregate" in set(stages["op"])
    assert any("aggregate" in t for t in res.trace)
    assert "WarpDB" not in repr(res) and "rows" in repr(res)


def test_estimate_and_explain(db):
    est = db.estimate("select g, sum(x) from t group by g")
    assert est["choice"] in ("cpu", "gpu")
    assert est["cpu_ms"] > 0 and est["gpu_ms"] > 0
    assert {"op", "cpu_ms", "gpu_ms", "rows_in", "rows_out"} <= set(est["stages"][0])
    text = db.explain("select g, sum(x) from t group by g")
    assert "decision:" in text and "TOTAL" in text
    analyzed = db.explain("select g, sum(x) from t group by g", analyze=True)
    assert "execution:" in analyzed


def test_auto_mode_logs_decisions_and_learns(db):
    n0 = len(db.decisions())
    db.sql("select count(*) from t", device="auto")
    d = db.decisions()
    assert len(d) == n0 + 1
    assert d.iloc[-1]["chosen"] in ("cpu", "gpu")
    m = db.cost_model_df()
    assert {"stage", "device", "a_ms", "b_ms_per_unit", "samples"} <= set(m.columns)
    assert db.cost_model()["hardware"]["h2d_gbs"] > 0


def test_gpu_residency_and_cache_control(db):
    db.drop_cache()
    r0 = db.residency()
    assert not r0[r0["table"] == "t"]["resident"].any()
    db.sql("select sum(x) from t where g = 3", device="gpu")
    r1 = db.residency()
    res = r1[(r1["table"] == "t") & r1["resident"]]["column"].tolist()
    assert set(res) == {"x", "g"}
    assert db.gpu_stats()["cache_mb"] > 0
    db.drop_cache()
    assert db.gpu_stats()["cache_mb"] == 0


def test_jit_toggle_and_stats(db):
    db.set_jit(True, min_rows=0)
    assert db.jit_enabled
    before = db.jit_stats()
    db.sql("select tag, sum(x * 2.5) from t where x > 1 group by tag", device="gpu")
    res = db.sql("select tag, sum(x * 2.5) from t where x > 1 group by tag", device="gpu")
    assert any("jit-fused" in t for t in res.trace)
    after = db.jit_stats()
    assert after["compiles"] + after["mem_hits"] + after["disk_hits"] > before["compiles"] + before["mem_hits"] + before["disk_hits"]
    db.set_jit(False)
    assert not db.jit_enabled
    assert not any("jit-fused" in t for t in db.sql("select tag, sum(x) from t group by tag", device="gpu").trace)
    db.set_jit(True, min_rows=0)


def test_errors_are_python_exceptions(db):
    with pytest.raises(warpdb.WarpError, match="column not found"):
        db.sql("select nope from t")
    with pytest.raises(warpdb.WarpError):
        db.sql("select from")
    with pytest.raises(ValueError):
        db.device = "tpu"


def test_cpu_only_database_never_touches_cuda(df):
    d = warpdb.connect(use_gpu=False)
    assert not d.has_gpu
    d.load_dataframe("t", df)
    assert d.sql("select count(*) as c from t").to_pandas()["c"][0] == len(df)
    with pytest.raises(RuntimeError):
        d.device = "gpu"


def test_query_catalog_matches_the_cpp_harness():
    import diff_test as dt

    norm = lambda s: " ".join(s.split())  # noqa: E731
    cpp = {norm(v) for v in dt.TPCH.values()}
    mine = {norm(v) for v in warpdb.queries.TPCH.values()}
    assert cpp == mine


@pytest.mark.skipif(not warpdb.tpch.available_scale_factors(), reason="no generated TPC-H data under data/")
def test_tpch_matches_duckdb():
    duckdb = pytest.importorskip("duckdb")
    sf = warpdb.tpch.available_scale_factors()[0]
    d = warpdb.connect()
    d.set_jit(True, min_rows=0)
    warpdb.tpch.load(d, sf, generate=False)
    con = duckdb.connect()
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    for name in ("Q1 pricing summary", "Q6 forecasting revenue", "Q14 promotion effect", "Q12 shipping modes"):
        sql = warpdb.queries.TPCH[name]
        want = con.execute(sql).fetchdf()
        for dev in ("cpu", "gpu"):
            got = d.sql(sql, device=dev).to_pandas()
            assert len(got) == len(want), (name, dev)
            for i, col in enumerate(got.columns):
                a, b = got[col].to_numpy(), want.iloc[:, i].to_numpy()
                if a.dtype.kind in "fiu":
                    np.testing.assert_allclose(a.astype(float), b.astype(float), rtol=1e-6, err_msg=f"{name} {dev} {col}")
                else:
                    assert [str(x) for x in a] == [str(x) for x in b], (name, dev, col)

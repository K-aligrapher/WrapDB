"""WarpDB dashboard - run SQL, see where and why it ran there, watch VRAM and the JIT.

    .\\.venv\\Scripts\\streamlit run python\\dashboard\\app.py
"""
import math
import os
import re
import statistics
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))
import warpdb  # noqa: E402
from warpdb import queries  # noqa: E402

st.set_page_config(page_title="WarpDB", page_icon="⚡", layout="wide")

CPU_C, GPU_C, ACCENT = "#4C78A8", "#59A14F", "#F28E2B"
OP_COLORS = {"scan": "#4C78A8", "join": "#F28E2B", "aggregate": "#59A14F", "sort": "#B07AA1", "project": "#9C755F",
             "upload": "#E15759"}


# --------------------------------------------------------------------------------------------- session & data
@st.cache_resource(show_spinner="Loading TPC-H into the engine ...")
def get_db(sf: float):
    db = warpdb.connect()
    warpdb.tpch.load(db, sf, generate=True)
    if db.has_gpu:
        db.set_jit(True, min_rows=200000)
    return db


@st.cache_resource(show_spinner="Preparing DuckDB reference ...")
def get_duck(sf: float):
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    return con


def log_history(db, label: str, res):
    st.session_state.setdefault("history", [])
    s = db.gpu_stats() if db.has_gpu else {}
    st.session_state.history.append(dict(
        n=len(st.session_state.history) + 1, query=label, device=res.device, ms=res.ms,
        cache_mb=s.get("cache_mb", 0.0), peak_mb=s.get("peak_mb", 0.0), in_use_mb=s.get("in_use_mb", 0.0)))


def timeline_fig(stages: pd.DataFrame) -> go.Figure:
    st_ = stages.copy()
    st_["order"] = st_["op"].map(lambda o: -1 if o == "upload" else 0)  # the PCIe upload happens before the compute
    st_ = st_.sort_values("order", kind="stable")
    fig = go.Figure()
    t = 0.0
    for _, s in st_.iterrows():
        name = f"{s.op} {s.label}".strip()
        fig.add_trace(go.Bar(
            y=[f"{name} ({s.device})"], x=[max(s.ms, 0.001)], base=[t], orientation="h",
            marker_color=OP_COLORS.get(s.op, "#888"), name=s.op, showlegend=False,
            hovertemplate=f"{name}<br>{s.ms:.3f} ms<br>rows {s.rows_in:,.0f} → {s.rows_out:,.0f}<extra></extra>"))
        t += max(s.ms, 0.001)
    fig.update_layout(height=60 + 38 * len(st_), margin=dict(l=0, r=10, t=10, b=30), xaxis_title="ms (cumulative)",
                      yaxis=dict(autorange="reversed"), bargap=0.25)
    return fig


def fmt_rows(x):
    return f"{x:,.0f}"


# --------------------------------------------------------------------------------------------- sidebar
st.sidebar.title("⚡ WarpDB")
sfs = warpdb.tpch.available_scale_factors() or [0.01]
sf = st.sidebar.selectbox("TPC-H scale factor", sfs, index=0, help="CSV data under data/ (generated on demand with DuckDB)")
db = get_db(sf)
if db.has_gpu:
    st.sidebar.success(f"GPU: {db.gpu_name}")
else:
    st.sidebar.warning("No CUDA device: CPU only")
tabs_info = db.tables()
st.sidebar.caption(" · ".join(f"{t['name']} {t['rows']:,}" for t in tabs_info if t["rows"] > 1000))

default_dev = "auto" if db.has_gpu else "cpu"
if db.has_gpu and not db.calibrated:
    if st.sidebar.button("Calibrate cost model (≈3 s)"):
        with st.spinner("Measuring PCIe, launch latency, operator throughput ..."):
            db.ensure_calibrated()
        st.rerun()
    st.sidebar.info("Auto placement calibrates itself on first use.")
elif db.has_gpu and st.sidebar.button("Re-calibrate"):
    with st.spinner("Calibrating ..."):
        db.calibrate()
    st.rerun()

hist = st.session_state.get("history", [])
st.sidebar.metric("Queries this session", len(hist))

st.title("WarpDB")
st.caption("From-scratch GPU SQL engine · hand-written kernels · NVRTC pipeline fusion · cost-based CPU/GPU placement")
tab_q, tab_opt, tab_gpu, tab_bench = st.tabs(["Query", "Optimizer", "GPU & JIT", "Benchmarks"])

# --------------------------------------------------------------------------------------------- Query tab
with tab_q:
    c1, c2, c3 = st.columns([3, 1, 1])
    preset = c1.selectbox("Example", list(queries.ALL), key="preset")
    device = c2.selectbox("Device", ["auto", "cpu", "gpu"] if db.has_gpu else ["cpu"], key="device")
    jit_on = c3.checkbox("JIT fusion", value=True, disabled=not db.has_gpu, key="jit")
    if st.session_state.get("_last_preset") != preset:
        st.session_state["sql"] = queries.one_line(queries.ALL[preset])
        st.session_state["_last_preset"] = preset
    sql = st.text_area("SQL", key="sql", height=130)
    run = st.button("▶ Run", type="primary")
    if db.has_gpu:
        db.set_jit(jit_on)
    if run:
        try:
            res = db.sql(sql, device=device)
            st.session_state["last"] = res
            log_history(db, preset, res)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.session_state.pop("last", None)
    res = st.session_state.get("last")
    if res is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Execution time", f"{res.ms:.2f} ms")
        m2.metric("Ran on", (res.device or "?").upper())
        m3.metric("Rows", f"{res.nrows:,}")
        fused = any("jit-fused" in t for t in res.trace)
        m4.metric("JIT-fused kernels", "yes" if fused else "no")
        if res.is_explain:
            st.code("\n".join(res.plan_lines), language="text")
        else:
            st.dataframe(res.to_pandas().head(1000), use_container_width=True)
        stg = res.stages
        if len(stg):
            st.subheader("Where the time went")
            st.plotly_chart(timeline_fig(stg), use_container_width=True)
        with st.expander("Optimizer decision" if res.plan_lines else "Optimizer decision (run with device = auto)"):
            st.code("\n".join(res.plan_lines) or "(not in auto mode)", language="text")
        with st.expander("Execution trace"):
            st.code("\n".join(res.trace), language="text")

# --------------------------------------------------------------------------------------------- Optimizer tab
with tab_opt:
    st.markdown("The optimizer prices every stage on both devices with a calibrated model `time = a + b·work`, adds the PCIe "
                "upload of columns that are not resident in VRAM, and picks the cheaper device. Estimates below are for the "
                "query in the **Query** tab.")
    sql_now = st.session_state.get("sql", queries.one_line(queries.ALL[next(iter(queries.ALL))]))
    if not db.has_gpu:
        st.info("No GPU: placement is CPU only.")
    try:
        est = db.estimate(sql_now)
    except Exception as e:  # noqa: BLE001
        est = None
        st.error(str(e))
    if est:
        cL, cR = st.columns([2, 1])
        stg = pd.DataFrame(est["stages"])
        stg["name"] = (stg["op"] + " " + stg["label"]).str.strip()
        fig = go.Figure()
        fig.add_bar(name="CPU", x=stg["name"], y=stg["cpu_ms"], marker_color=CPU_C)
        fig.add_bar(name="GPU", x=stg["name"], y=stg["gpu_ms"], marker_color=GPU_C)
        if est["upload_ms"] > 0:
            fig.add_bar(name="GPU upload (PCIe)", x=["upload"], y=[est["upload_ms"]], marker_color=OP_COLORS["upload"])
        fig.update_layout(barmode="group", height=340, yaxis_title="estimated ms", margin=dict(t=20, b=20))
        cL.plotly_chart(fig, use_container_width=True)
        with cR:
            choice = est["choice"].upper()
            st.metric("Decision", choice)
            st.metric("Estimated CPU", f"{est['cpu_ms']:.2f} ms")
            st.metric("Estimated GPU", f"{est['gpu_ms']:.2f} ms", help="includes PCIe upload of cold columns")
            st.caption(f"{est['cols_cold']}/{est['cols_total']} referenced columns are cold in VRAM "
                       f"({est['upload_mb']:.1f} MB to upload)")
            if not est["gpu_feasible"]:
                st.warning("GPU not feasible: " + est["gpu_reason"])
        with st.expander("Per-stage table"):
            st.dataframe(stg[["name", "rows_in", "rows_aux", "rows_out", "cpu_ms", "gpu_ms"]], use_container_width=True)

        st.subheader("What if? Run it on both devices")
        wc1, wc2 = st.columns([1, 3])
        cold = wc1.checkbox("Cold GPU (empty VRAM cache)", value=False)
        if wc2.button("Measure CPU vs GPU and compare with the estimate", disabled=not db.has_gpu):
            try:
                db.sql(sql_now, device="cpu")
                if cold:
                    db.drop_cache()
                else:
                    db.sql(sql_now, device="gpu")
                cpu_runs = [db.sql(sql_now, device="cpu").ms for _ in range(3)]
                gpu_ms = []
                for _ in range(3):
                    if cold:
                        db.drop_cache()
                    gpu_ms.append(db.sql(sql_now, device="gpu").ms)
                cpu_a, gpu_a = statistics.median(cpu_runs), statistics.median(gpu_ms)
                best = "GPU" if gpu_a < cpu_a else "CPU"
                ok = best == choice
                st.dataframe(pd.DataFrame({"": ["estimated", "measured"], "CPU ms": [est["cpu_ms"], cpu_a],
                                           "GPU ms": [est["gpu_ms"], gpu_a]}), hide_index=True)
                (st.success if ok else st.error)(f"Measured winner: {best}. The optimizer chose {choice}: "
                                                 f"{'correct' if ok else 'a miss'}.")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

    st.subheader("Cost model")
    cm = db.cost_model()
    hw = cm["hardware"]
    if hw["has_gpu"]:
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("PCIe host→device", f"{hw['h2d_gbs']:.1f} GB/s")
        h2.metric("PCIe device→host", f"{hw['d2h_gbs']:.1f} GB/s")
        h3.metric("Kernel launch + sync", f"{hw['launch_us']:.0f} µs")
        h4.metric("VRAM copy", f"{hw['gpu_membw_gbs']:.0f} GB/s")
    mdf = db.cost_model_df()
    mdf = mdf[(mdf["samples"] > 0) | (mdf["device"] == "cpu")]
    st.dataframe(mdf, use_container_width=True, hide_index=True)

    dec = db.decisions()
    st.subheader("Decision log (auto mode)")
    if len(dec) == 0:
        st.caption("Run queries with Device = auto to populate the log.")
    else:
        dec["est_chosen_ms"] = np.where(dec["chosen"] == "gpu", dec["est_gpu_ms"], dec["est_cpu_ms"])
        fig = go.Figure()
        for dev, col in (("cpu", CPU_C), ("gpu", GPU_C)):
            d = dec[dec["chosen"] == dev]
            fig.add_scatter(x=d["est_chosen_ms"], y=d["actual_ms"], mode="markers", name=f"ran on {dev}",
                            marker=dict(color=col, size=9), text=d["sql"].str.slice(0, 60))
        lo = max(0.01, min(dec["est_chosen_ms"].min(), dec["actual_ms"].min()) * 0.7)
        hi = max(dec["est_chosen_ms"].max(), dec["actual_ms"].max()) * 1.4
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="perfect")
        fig.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="predicted ms (chosen device)",
                          yaxis_title="actual ms", height=360, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dec.drop(columns=["est_chosen_ms"]).tail(50), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------------- GPU & JIT tab
with tab_gpu:
    if not db.has_gpu:
        st.info("No CUDA device available.")
    else:
        s = db.gpu_stats()
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Column cache", f"{s['cache_mb']:.0f} / {s['cache_budget_mb']:.0f} MB")
        g2.metric("Allocator in use", f"{s['in_use_mb']:.0f} MB")
        g3.metric("Peak (last query)", f"{s['peak_mb']:.0f} MB")
        g4.metric("Uploaded in total", f"{s['uploaded_total_mb']:.0f} MB")
        st.progress(min(1.0, s["cache_mb"] / max(s["cache_budget_mb"], 1)), text="VRAM column-cache budget")
        if st.button("Evict all columns from VRAM"):
            db.drop_cache()
            st.rerun()

        rdf = db.residency()
        rdf["res_mb"] = np.where(rdf["resident"], rdf["mb"], 0.0)
        by_t = rdf.groupby("table", as_index=False).agg(**{"resident MB": ("res_mb", "sum"), "total MB": ("mb", "sum")})
        fig = go.Figure()
        fig.add_bar(x=by_t["table"], y=by_t["resident MB"], name="resident in VRAM", marker_color=GPU_C)
        fig.add_bar(x=by_t["table"], y=by_t["total MB"] - by_t["resident MB"], name="host only", marker_color="#BBBBBB")
        fig.update_layout(barmode="stack", height=300, yaxis_title="MB", margin=dict(t=20))
        cA, cB = st.columns([1, 1])
        cA.plotly_chart(fig, use_container_width=True)
        with cB.expander("Column-level residency", expanded=False):
            st.dataframe(rdf.sort_values(["table", "column"]), use_container_width=True, hide_index=True, height=300)

        if hist:
            hdf = pd.DataFrame(hist)
            fig = go.Figure()
            fig.add_scatter(x=hdf["n"], y=hdf["cache_mb"], name="column cache MB", mode="lines+markers", line_color=GPU_C)
            fig.add_scatter(x=hdf["n"], y=hdf["peak_mb"], name="allocator peak MB", mode="lines+markers", line_color=ACCENT)
            fig.update_layout(height=280, xaxis_title="query #", yaxis_title="MB", margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("JIT pipeline fusion")
        js = db.jit_stats()
        j1, j2, j3, j4 = st.columns(4)
        j1.metric("NVRTC compiles", f"{js['compiles']}", help=f"{js['compile_ms']:.0f} ms total")
        j2.metric("In-memory cache hits", js["mem_hits"])
        j3.metric("On-disk cache hits", js["disk_hits"], help="CUBIN reused from an earlier run: no NVRTC")
        j4.metric("Failures (fell back)", js["failures"])
        st.caption("Kernels are compiled once (100s of ms), cached in memory and on disk, then reused by every query of "
                   "the same shape.")
        ab_q = st.selectbox("Fused vs interpreted", list(queries.ALL), key="ab_q")
        ab_n = st.slider("Runs", 3, 25, 9, key="ab_n")
        if st.button("Measure fusion speedup"):
            q = queries.ALL[ab_q]
            db.set_jit(True, min_rows=0)
            for mode in (False, True, False, True):
                db.set_jit(mode)
                db.sql(q, device="gpu")
            u, f = [], []
            for _ in range(ab_n):
                for mode, bucket in ((False, u), (True, f)):
                    db.set_jit(mode)
                    bucket.append(db.sql(q, device="gpu").ms)
            db.set_jit(st.session_state.get("jit", True))
            mu, mf = statistics.median(u), statistics.median(f)
            fig = go.Figure(go.Bar(x=["interpreted (unfused)", "JIT-fused"], y=[mu, mf], marker_color=[CPU_C, GPU_C],
                                   text=[f"{mu:.2f} ms", f"{mf:.2f} ms"], textposition="outside"))
            fig.update_layout(height=300, yaxis_title="ms (median)", margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Fusion speedup: {mu / mf:.2f}x")

# --------------------------------------------------------------------------------------------- Benchmarks tab
with tab_bench:
    st.markdown(f"Live comparison on TPC-H SF{sf:g}. *GPU warm* = columns resident in VRAM, *GPU cold* = VRAM cache emptied "
                "before every run (includes the PCIe upload).")
    names = list(queries.TPCH)
    sel = st.multiselect("Queries", names, default=names)
    engines = st.multiselect("Engines", ["WarpDB CPU", "WarpDB GPU warm", "WarpDB GPU cold", "WarpDB auto", "DuckDB"],
                             default=["WarpDB CPU", "WarpDB GPU warm", "WarpDB auto", "DuckDB"] if db.has_gpu else ["WarpDB CPU", "DuckDB"])
    reps = st.slider("Repetitions (median)", 1, 15, 5, key="reps")
    if st.button("Run benchmark", key="bench"):
        rows = []
        bar = st.progress(0.0)
        total = max(1, len(sel) * len(engines))
        done = 0
        con = get_duck(sf) if "DuckDB" in engines else None
        for q in sel:
            sql = queries.TPCH[q]
            one = queries.one_line(sql)
            for eng in engines:
                xs = []
                try:
                    if eng == "WarpDB CPU":
                        db.sql(one, device="cpu")
                        xs = [db.sql(one, device="cpu").ms for _ in range(reps)]
                    elif eng == "WarpDB GPU warm":
                        db.sql(one, device="gpu")
                        xs = [db.sql(one, device="gpu").ms for _ in range(reps)]
                    elif eng == "WarpDB GPU cold":
                        for _ in range(reps):
                            db.drop_cache()
                            t = time.perf_counter()
                            db.sql(one, device="gpu")
                            xs.append((time.perf_counter() - t) * 1000)
                    elif eng == "WarpDB auto":
                        db.sql(one, device="auto")
                        xs = [db.sql(one, device="auto").ms for _ in range(reps)]
                    elif eng == "DuckDB":
                        con.execute(sql).fetchall()
                        for _ in range(reps):
                            t = time.perf_counter()
                            con.execute(sql).fetchall()
                            xs.append((time.perf_counter() - t) * 1000)
                    rows.append(dict(query=q.split()[0], engine=eng, ms=statistics.median(xs)))
                except Exception as e:  # noqa: BLE001
                    st.warning(f"{q} on {eng}: {e}")
                done += 1
                bar.progress(done / total)
        st.session_state["bench_rows"] = pd.DataFrame(rows)
    bdf = st.session_state.get("bench_rows")
    if bdf is not None and len(bdf):
        palette = {"WarpDB CPU": CPU_C, "WarpDB GPU warm": GPU_C, "WarpDB GPU cold": "#8CD17D", "WarpDB auto": ACCENT,
                   "DuckDB": "#B6992D"}
        fig = go.Figure()
        for eng in bdf["engine"].unique():
            d = bdf[bdf["engine"] == eng]
            fig.add_bar(name=eng, x=d["query"], y=d["ms"], marker_color=palette.get(eng))
        fig.update_layout(barmode="group", yaxis_type="log", yaxis_title="ms (median, log)", height=380, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        piv = bdf.pivot(index="query", columns="engine", values="ms")
        gm = piv.apply(lambda c: math.exp(np.log(c.dropna()).mean()) if c.notna().any() else np.nan)
        piv.loc["geomean"] = gm
        st.dataframe(piv.round(2), use_container_width=True)
        if "DuckDB" in piv.columns:
            sp = (piv["DuckDB"].values[:, None] / piv.drop(columns=["DuckDB"]).values)
            sp_df = pd.DataFrame(sp, index=piv.index, columns=[f"{c} vs DuckDB" for c in piv.drop(columns=["DuckDB"]).columns])
            st.caption("Speedup over DuckDB (>1 = WarpDB is faster)")
            st.dataframe(sp_df.round(2), use_container_width=True)

    for doc, title in (("BENCHMARKS_TPCH.md", "Recorded TPC-H run"), ("BENCHMARKS_JIT.md", "Recorded JIT fusion run"),
                       ("PLACEMENT_EVAL.md", "Recorded placement evaluation")):
        path = os.path.join(ROOT, "docs", doc)
        if os.path.exists(path):
            with st.expander(f"{title} (docs/{doc})"):
                st.markdown(open(path, encoding="utf8").read().split("## Per query")[0])

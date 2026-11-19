"""Headless smoke tests of the Streamlit dashboard (streamlit.testing.AppTest): every tab renders, queries run,
the optimizer view works - with no exceptions raised inside the app script.

Run:  .\\.venv\\Scripts\\python -m pytest tests/test_dashboard.py -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

pytest.importorskip("warpdb", reason="build the extension first: .\\build.ps1")
pytest.importorskip("streamlit")
import warpdb  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(ROOT, "python", "dashboard", "app.py")
pytestmark = pytest.mark.skipif(not warpdb.tpch.available_scale_factors(), reason="needs generated TPC-H data under data/")


def _button(at, label_part):
    for b in at.button:
        if label_part in b.label:
            return b
    raise AssertionError(f"no button containing {label_part!r}: {[b.label for b in at.button]}")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_app_renders_all_tabs_without_exceptions(app):
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.tabs) == 4
    assert [t.label for t in app.tabs] == ["Query", "Optimizer", "GPU & JIT", "Benchmarks"]


def test_running_a_query_shows_results_and_metrics(app):
    app.selectbox(key="device").select("cpu")
    _button(app, "Run").click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    labels = [m.label for m in app.metric]
    assert "Execution time" in labels and "Ran on" in labels
    ran_on = [m.value for m in app.metric if m.label == "Ran on"][0]
    assert ran_on == "CPU"
    assert len(app.dataframe) >= 1


def test_gpu_query_and_optimizer_tab(app):
    if not warpdb.connect().has_gpu:
        pytest.skip("no GPU")
    app.selectbox(key="device").select("gpu")
    _button(app, "Run").click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert [m.value for m in app.metric if m.label == "Ran on"][0] == "GPU"
    # the optimizer tab estimated the same query
    assert "Decision" in [m.label for m in app.metric]
    assert "Estimated CPU" in [m.label for m in app.metric]


def test_invalid_sql_is_reported_not_raised(app):
    app.text_area(key="sql").set_value("select nope from lineitem")
    _button(app, "Run").click()
    app.run()
    assert not app.exception
    assert any("not found" in e.value for e in app.error)


def test_benchmark_tab_runs_and_charts(app):
    ms = [m for m in app.multiselect if m.label == "Queries"][0]
    ms.set_value(["Q6 forecasting revenue", "Q14 promotion effect"])
    app.slider(key="reps").set_value(1)
    _button(app, "Run benchmark").click()
    app.run(timeout=300)
    assert not app.exception, [e.value for e in app.exception]
    assert not [w for w in app.warning if "on WarpDB" in w.value or "on DuckDB" in w.value], [w.value for w in app.warning]
    frames = [d.value for d in app.dataframe if "geomean" in list(d.value.index)]
    assert frames, "benchmark table with a geomean row expected"
    assert {"Q6", "Q14", "geomean"} <= set(frames[0].index)

"""Pythonic wrapper over the native engine: pandas results, TPC-H helpers, optimizer/GPU introspection."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import _warpdb


class Result:
    """A query result. `to_pandas()` materializes it; the metadata describes how it was computed."""

    def __init__(self, raw: "_warpdb.Result", sql: str = ""):
        self._raw = raw
        self.sql = sql
        self._df: Optional[pd.DataFrame] = None

    # ---- metadata ----
    @property
    def ms(self) -> float:
        """Engine execution time in milliseconds (excludes the Python <-> C++ conversion)."""
        return self._raw.ms

    @property
    def device(self) -> str:
        """The device the query ran on: 'cpu' or 'gpu' (the optimizer's choice in auto mode)."""
        return self._raw.device

    @property
    def nrows(self) -> int:
        return self._raw.nrows

    @property
    def names(self) -> List[str]:
        return list(self._raw.names)

    @property
    def trace(self) -> List[str]:
        """Per-stage execution trace lines."""
        return list(self._raw.trace)

    @property
    def plan_lines(self) -> List[str]:
        """The optimizer's placement estimate (auto mode / EXPLAIN)."""
        return list(self._raw.plan_lines)

    @property
    def is_explain(self) -> bool:
        return self._raw.is_explain

    @property
    def stages(self) -> pd.DataFrame:
        """Measured stages: op, device, label, rows in/out, time, work."""
        return pd.DataFrame(self._raw.stages())

    # ---- data ----
    def to_pandas(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        data: Dict[str, Any] = {}
        for c in self._raw.columns():
            kind, arr = c["kind"], c["data"]
            if kind == "date":
                col = pd.to_datetime(arr, unit="D")
            elif kind == "str":
                col = pd.Categorical.from_codes(arr, categories=c["categories"])
                col = pd.Series(col).astype(object)
            else:
                col = arr
            name = c["name"]
            while name in data:  # duplicate output names are legal in SQL, not in a DataFrame
                name += "_"
            data[name] = col
        self._df = pd.DataFrame(data)
        return self._df

    def __len__(self) -> int:
        return self.nrows

    def __repr__(self) -> str:
        return f"<warpdb.Result {self.nrows} rows x {len(self.names)} cols, {self.ms:.2f} ms on {self.device or '?'}>"

    def _repr_html_(self):  # Jupyter
        return self.to_pandas()._repr_html_()


class Database:
    def __init__(self, use_gpu: bool = True):
        self._db = _warpdb.Database(use_gpu)

    # ---- data ----
    def load_csv(self, name: str, path: str, delimiter: str = ",", header: bool = True) -> Dict[str, Any]:
        return self._db.load_csv(name, os.fspath(path), delimiter, header)

    def load_tpch(self, sf: float = 1, generate: bool = True) -> Dict[str, Dict[str, Any]]:
        from . import tpch

        return tpch.load(self, sf, generate=generate)

    def load_dataframe(self, name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Load a pandas DataFrame (goes through a temporary CSV: fine for tests and notebooks)."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"{name}.csv")
            df.to_csv(path, index=False)
            return self.load_csv(name, path)

    def tables(self) -> List[Dict[str, Any]]:
        return self._db.tables()

    # ---- queries ----
    def sql(self, query: str, device: Optional[str] = None) -> Result:
        """Run a query. `device` overrides the session device for this one query ('cpu' | 'gpu' | 'auto')."""
        return Result(self._db.execute(query, device or ""), query)

    def estimate(self, query: str) -> Dict[str, Any]:
        """The optimizer's per-stage CPU/GPU cost estimate and decision, without executing the query."""
        return self._db.estimate(query)

    def explain(self, query: str, analyze: bool = False) -> str:
        res = self.sql(("explain analyze " if analyze else "explain ") + query)
        return "\n".join(res.plan_lines + (["execution:"] + ["  " + t for t in res.trace] if analyze else []))

    # ---- session ----
    @property
    def device(self) -> str:
        return self._db.device

    @device.setter
    def device(self, d: str) -> None:
        self._db.device = d

    @property
    def has_gpu(self) -> bool:
        return self._db.has_gpu

    @property
    def gpu_name(self) -> str:
        return self._db.gpu_name

    @property
    def learn(self) -> bool:
        return self._db.learn

    @learn.setter
    def learn(self, b: bool) -> None:
        self._db.learn = b

    # ---- optimizer ----
    def calibrate(self) -> None:
        self._db.calibrate()

    def ensure_calibrated(self) -> None:
        self._db.ensure_calibrated()

    @property
    def calibrated(self) -> bool:
        return self._db.calibrated

    def cost_model(self) -> Dict[str, Any]:
        return self._db.cost_model()

    def cost_model_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._db.cost_model()["models"])

    def decisions(self) -> pd.DataFrame:
        return pd.DataFrame(self._db.decisions())

    # ---- GPU ----
    def gpu_stats(self) -> Dict[str, float]:
        return self._db.gpu_stats()

    def residency(self) -> pd.DataFrame:
        return pd.DataFrame(self._db.residency())

    def drop_cache(self) -> None:
        self._db.drop_cache()

    # ---- JIT ----
    def set_jit(self, on: bool, min_rows: Optional[int] = None) -> None:
        self._db.set_jit(on)
        if min_rows is not None:
            self._db.set_jit_min_rows(int(min_rows))

    @property
    def jit_enabled(self) -> bool:
        return self._db.jit_enabled

    def jit_stats(self) -> Dict[str, float]:
        return self._db.jit_stats()

    def jit_preload(self, max_kernels: int = 64) -> int:
        """Load recently used fused kernels from the on-disk cache now, instead of during the first query."""
        return self._db.jit_preload(max_kernels)


def connect(use_gpu: bool = True) -> Database:
    """Create an in-process database. With use_gpu=False the CUDA backend is not initialized at all."""
    return Database(use_gpu)

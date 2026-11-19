"""WarpDB - a GPU-accelerated SQL engine.

    import warpdb
    db = warpdb.connect()                       # CPU + GPU, cost-based placement
    db.load_csv("t", "data.csv")
    res = db.sql("select g, sum(x) from t group by g")
    res.to_pandas(), res.ms, res.device, res.stages

The native extension (`_warpdb`) is built by CMake into this directory (see build.ps1).
"""
import os
import shutil
import sys


def _add_cuda_dll_dirs():
    """Windows does not search PATH for the DLLs an extension module imports (since Python 3.8): NVRTC and the
    CUDA runtime must be made findable explicitly."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    candidates = []
    for var in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(var):
            candidates += [os.path.join(os.environ[var], "bin"), os.path.join(os.environ[var], "bin", "x64")]
    nvcc = shutil.which("nvcc")
    if nvcc:
        d = os.path.dirname(nvcc)
        candidates += [d, os.path.join(d, "x64")]
    for d in candidates:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


_add_cuda_dll_dirs()

from . import _warpdb  # noqa: E402,F401
from ._warpdb import WarpError  # noqa: E402,F401
from .api import Database, Result, connect  # noqa: E402,F401
from . import queries, tpch  # noqa: E402,F401

__all__ = ["connect", "Database", "Result", "WarpError", "queries", "tpch"]

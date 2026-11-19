# WarpDB

A from-scratch GPU SQL engine (C++/CUDA, no libcudf) with JIT-fused pipelines and a
residency-aware CPU/GPU placement optimizer. See [docs/PRD.md](docs/PRD.md).

**Status:** Phase 2 — hand-written GPU primitives done and benchmarked against CUB
([docs/BENCHMARKS_GPU_PRIMS.md](docs/BENCHMARKS_GPU_PRIMS.md)): single-pass decoupled look-back scan (parity with CUB),
fused warp-ballot compaction (parity or better, 3-5x faster than flags+scan+scatter), stable onesweep-style radix sort
(0.65-0.86x of CUB), lock-free open-addressing hash table (7-10x faster than sort-based distinct counting).
Run `.\build\release\warpdb_bench.exe`. Kernels live in `src/gpu/prims/`.

**Phase 1** — CPU engine complete: columnar storage, CSV ingest, SQL parser, vectorized multithreaded
executor (filter / hash join / hash aggregate / sort / top-k), verified against DuckDB on TPC-H.

```powershell
.\build\release\warpdb.exe
warpdb> .load lineitem data\tpch_sf1\lineitem.csv
warpdb> EXPLAIN SELECT l_returnflag, sum(l_quantity) FROM lineitem GROUP BY l_returnflag;
```

Differential tests vs DuckDB (generates TPC-H data on first run, 31 queries):
`.\.venv\Scripts\python tests\diff_test.py --sf 0.1`

Known v1 limits: no NULLs, inner equi-joins only, DECIMAL stored as DOUBLE, no subqueries / window functions.

## Build (native Windows)

Prerequisites: Visual Studio 2022/2026 (C++ workload), Windows SDK, CUDA Toolkit 13.x, Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\build.ps1 -Test          # configure + build + run tests
.\build\release\warpdb.exe # prints GPU info and runs the hello kernel
```

Notes: CMake is pinned to 3.31 (4.4.5 mis-generated Ninja rules in our setup). `-Clean` wipes the build dir.

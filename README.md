# WarpDB

A from-scratch GPU SQL engine (C++/CUDA, no libcudf) with JIT-fused pipelines and a
residency-aware CPU/GPU placement optimizer. See [docs/PRD.md](docs/PRD.md).

**Status:** Phase 4 — cost-based CPU/GPU placement. `.device auto` picks, per query, the device with the lower
predicted cost: a calibrated model `time = a + b * work` per (stage, device), cardinalities from sampling and NDV
sketches, and the PCIe upload cost of every column that is not already resident in VRAM. `EXPLAIN <query>` prints the
per-stage estimate for both devices and the decision without executing; `EXPLAIN ANALYZE` also runs it. The model is
calibrated once per machine (~3 s, cached in `%LOCALAPPDATA%\warpdb\profile.json`) and keeps learning from every query.
Evaluation against an oracle (best of forced CPU / forced GPU) on TPC-H SF0.01 / 0.1 / 1, warm and cold
([docs/PLACEMENT_EVAL.md](docs/PLACEMENT_EVAL.md)): **95% hit rate, 97% on high-stakes decisions, 100% at SF1 with
8-9% mean regret**; a whole workload runs 8.9x faster than always-CPU and 16% faster than always-GPU.
`.\.venv\Scripts\python tests\eval_placement.py`. CLI: `.calibrate`, `.model`, `.decisions`.

**Phase 3** — SQL executes on the GPU. `.device gpu` in the CLI runs the same queries with CUDA kernels
(scan+filter, hash join, hash aggregate, radix sort, top-k) on top of a caching GPU allocator and a VRAM column cache.
All 31 differential queries match DuckDB on the GPU at SF0.1 and SF1. TPC-H SF1 (warm, columns resident in VRAM):
geomean **7.1 ms** on the GPU vs 190 ms on the CPU path (26.7x) vs 54 ms for DuckDB (7.6x) —
see [docs/BENCHMARKS_TPCH.md](docs/BENCHMARKS_TPCH.md). Cold runs include the PCIe upload of the needed columns.
`.\.venv\Scripts\python tests\diff_test.py --sf 1 --device gpu`, `.\.venv\Scripts\python tests\bench_tpch.py`.

**Phase 2** — hand-written GPU primitives done and benchmarked against CUB
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

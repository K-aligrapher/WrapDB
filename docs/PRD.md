# PRD — WarpDB: a from-scratch GPU SQL engine with a residency-aware hybrid optimizer

*Working name. Status: Draft v0.1 · 2026-11-19*

---

## 1. One-line pitch

A small analytical SQL engine, written in C++/CUDA with no libcudf, that runs filters, joins, aggregations and sorts in **hand-written GPU kernels**. Its own **JIT kernel compiler** fuses whole query pipelines into a single kernel. A **learned, operator-level CPU/GPU placement optimizer** accounts for where the data currently lives and what the PCIe transfer costs. Every query can be inspected in a live "flight recorder" dashboard.

## 2. Why this project, and why this version of it

### 2.1 The landscape today (Nov 2026)

| System | What it is | What it tells us |
|---|---|---|
| **Sirius** (UW-Madison + NVIDIA, CIDR 2026) | GPU-native SQL engine plugged into DuckDB via Substrait; operators come from libcudf | "GPU SQL engine" is now a published research system. Copying it is not novel. |
| **Velox / Presto + cuDF**, **cudf-polars**, NVIDIA **GQE** | Production engines that call libcudf for joins/aggs/sorts | The industry trend is *don't write kernels, call cuDF*, so writing them yourself is now the rare skill. |
| **HeavyDB** (ex-OmniSci/MapD) | Mature GPU DB with LLVM JIT | Reportedly abandoned after NVIDIA's 2025 acquisition. JIT-compiled GPU queries are proven valuable. |
| **CoGaDB + HyPE** (Breß et al., 2013–2017) | Learned regression cost models for CPU/GPU operator placement | **The "adaptive CPU-vs-GPU optimizer" in the original idea already exists.** HyPE has no explicit data-residency or transfer-overlap modelling and no multi-query awareness. |
| **Crystal** (SIGMOD 2020) | Tile-based GPU query routines; avoids materialization between operators | Fusion and tile-based execution give large speedups. It is a hand-written library, not a compiler. |
| **Wu et al.** (SIGMOD 2025) | Radix-sort-based partitioning for GPU joins/group-by, up to 2.3× | Gives us a modern, citable kernel design to implement and benchmark. |
| **Mordred** (PVLDB 2022) | Cost-based decision on which data to cache in GPU memory | Residency matters. This motivates our residency-aware placement. |

### 2.2 Our positioning

A plain "GPU executes SQL" project is a demo. Interviewers who know the space will ask "why not cuDF?" and "isn't that HyPE?" This version has answers to both:

1. **No libcudf.** Every core primitive (scan, compaction, radix sort, hash table, hash join, group-by) is our own kernel. Each one is benchmarked against CUB/Thrust so we can say *"my radix sort is within X% of CUB."*
2. **JIT pipeline fusion via NVRTC.** Each query's filter→project→aggregate pipeline is compiled into one CUDA kernel at runtime. Crystal does this by hand; we generate it. We measure fused against operator-at-a-time execution.
3. **Residency- and transfer-aware hybrid placement.** HyPE learns per-operator cost. We also model (a) whether each column is already in VRAM, (b) PCIe cost with overlap from async streams, and (c) the cost of switching devices mid-plan. Placement is per operator, not per query.
4. **Larger-than-VRAM execution on a laptop.** With 8 GB VRAM we must stream chunks (pinned memory, double-buffered async copies) and use lightweight compression to shrink transfers.
5. **Differential fuzz testing against DuckDB.** Thousands of random queries are checked for identical results. Most student engines are never verified like this.
6. **Flight-recorder dashboard.** It shows the plan DAG coloured by device, predicted vs actual cost per operator, a CUDA timeline of kernels and copies, and the optimizer's reasoning ("GPU rejected: 1.2 GB transfer > 40 ms CPU scan").

## 3. Goals and non-goals

### Goals
- G1: Run a defined analytical SQL subset (§5) correctly on CPU and GPU, verified against DuckDB.
- G2: GPU path beats our own multithreaded CPU path by ≥10× on scan/aggregate-heavy SSB queries at SF10, and is competitive with DuckDB (target: within 2× of DuckDB, faster on some queries).
- G3: The placement optimizer picks the faster plan in ≥90% of benchmark queries, with ≤10% average regret vs. an oracle that tries every placement.
- G4: Queries over data larger than free VRAM complete via streaming.
- G5: Anyone can see why each query ran where it did, using the dashboard.

### Non-goals
- Transactions, UPDATE/DELETE, indexes, durability/WAL, concurrency control.
- Full SQL (subqueries, window functions, outer joins) in v1. Some are listed as stretch goals.
- Multi-GPU or distributed execution.
- Using libcudf or another GPU DB library in the engine core. CUB/Thrust are used **only as benchmark baselines**.

## 4. Target environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 5050 Laptop, Blackwell, compute capability 12.0 (`sm_120`), 8 GB VRAM |
| Toolkit | CUDA 13.2 (installed), NVRTC for JIT |
| OS | Windows 11. **Open question:** native MSVC vs WSL2 (see §13) |
| Build | CMake ≥3.28 + Ninja (**not yet installed**), C++20 |
| Python | 3.10 for dashboard, bindings (pybind11), benchmark harness |
| Baselines | DuckDB (Python package), SQLite, Polars. cudf-polars only if on WSL2. |

## 5. SQL scope

**v1 (must):**
```sql
SELECT <expr list | aggregates>
FROM t1 [JOIN t2 ON t1.a = t2.b]...      -- inner equi-joins, star-schema shaped
WHERE <predicate>                         -- =,<>,<,<=,>,>=, AND/OR/NOT, BETWEEN, IN (list)
GROUP BY <cols>
HAVING <predicate on aggregates>
ORDER BY <cols> [ASC|DESC]
LIMIT n
```
- Aggregates: `COUNT(*)`, `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`.
- Expressions: arithmetic (`+ - * /`), `CASE WHEN`, `EXTRACT(YEAR FROM date)`.
- Types: `INT32`, `INT64`, `FLOAT64`, `DATE` (int32 days), `DECIMAL(p,s)` (scaled int64), `VARCHAR` (dictionary-encoded).
- Meta commands: `EXPLAIN`, `EXPLAIN ANALYZE`, `SET device = auto|cpu|gpu`, `.load csv`, `.import sqlite`.

**v2 (stretch):** `COUNT(DISTINCT)`, `LIKE 'prefix%'` on dictionaries, `LEFT JOIN`, uncorrelated `IN (subquery)`, top-K for `ORDER BY ... LIMIT`.

This covers all 13 **Star Schema Benchmark (SSB)** queries and a TPC-H subset (Q1, Q3, Q5, Q6, Q10, Q12, Q14).

## 6. System architecture

```
 SQL text
   │
   ▼
┌──────────┐   ┌─────────┐   ┌─────────────────┐   ┌──────────────────────────┐
│  Lexer + │──▶│ Binder  │──▶│ Logical plan +  │──▶│  Placement optimizer     │
│  Parser  │   │(catalog,│   │ rule rewrites   │   │  (cost model + residency │
│ (hand-   │   │ types)  │   │ pushdown,       │   │   + transfer, DP over    │
│ written) │   └─────────┘   │ pruning, join   │   │   plan with device state)│
└──────────┘                 │ order (greedy)  │   └────────────┬─────────────┘
                             └─────────────────┘                │ physical plan
                                                                ▼
                      ┌──────────────────────────────────────────────────┐
                      │                Executor (pipelines)              │
                      │  ┌──────────────┐          ┌───────────────────┐ │
                      │  │ CPU backend  │◀─xfer───▶│ GPU backend       │ │
                      │  │ vectorized,  │  async   │ • kernel library  │ │
                      │  │ multithreaded│  streams │ • JIT fused kernels│ │
                      │  └──────────────┘          │ • memory manager  │ │
                      │                            └───────────────────┘ │
                      └──────────────────────────────────────────────────┘
                                         │ telemetry (JSON / shared buffer)
                                         ▼
                       Python: bindings · benchmark harness · dashboard
```

### 6.1 Storage layer
- **Columnar in-memory tables** with chunks of 1–4 M rows ("row groups") and per-chunk **zone maps** (min/max) so filters can skip whole chunks.
- **Ingest:** CSV (parallel parser), SQLite import (via sqlite3 C API), and our own binary `.wcol` format for fast reloads.
- **Encodings:** dictionary for strings, frame-of-reference + bit-packing for integers. The GPU can decode in-kernel, and filters on dictionary columns compare codes rather than strings.

### 6.2 Query front end
- Hand-written recursive-descent parser. It produces an AST with good error messages (line/column, caret).
- Binder resolves names and types against the catalog.
- Rule-based rewrites: predicate pushdown, projection pruning, constant folding, join reordering (greedy, smallest build side first, star-schema aware).

### 6.3 GPU backend — kernel library (built from scratch)

| Primitive | Implementation plan | Baseline to beat / match |
|---|---|---|
| **Prefix scan** | Decoupled look-back single-pass scan (Merrill & Garland) | `cub::DeviceScan` |
| **Stream compaction** (filter) | Predicate → flags → scan → scatter. Then warp-ballot (`__ballot_sync` + `__popc`) variant | `thrust::copy_if` |
| **Radix sort** | LSD radix sort, 8-bit digits, onesweep-style. Key-value for ORDER BY, multi-column via packed keys | `cub::DeviceRadixSort` |
| **Hash table** | Open addressing, linear probing, `atomicCAS`, power-of-two capacity, SoA layout | cuCollections `static_map` (reference only) |
| **Hash aggregation** | Two-level: shared-memory per-block tables for low-cardinality groups, global table fallback for high cardinality. Chosen by estimated NDV | sort-based group-by |
| **Hash join** | Build/probe; plus radix-partitioned join (Wu et al. 2025) for large builds | non-partitioned version |
| **Top-K** | Per-block heap + merge, for `ORDER BY ... LIMIT` | full sort |
| **Decompression** | Bit-unpack / FOR decode fused into scan | uncompressed read |

### 6.4 GPU backend — JIT fusion (headline feature #1)
- Pipelines are split at **pipeline breakers** (hash build, aggregation, sort).
- Each non-breaking pipeline (scan → decode → filter → project → probe → partial-aggregate) is emitted as **CUDA C++ source** and compiled with **NVRTC**, then cached by plan hash.
- Generated kernels are tile-based (Crystal-style): each thread block loads a tile into registers/shared memory and applies every operator before writing anything.
- **Fallback:** an interpreted operator-at-a-time path using the precompiled kernel library. Every query runs both ways in tests, and benchmarks report fused vs. unfused speedup and JIT compile time (amortized via the cache).

### 6.5 GPU memory manager
- One `cudaMallocAsync` pool (stream-ordered allocator) with our own tracking layer: per-query arena, high-water marks, OOM detection.
- **Column cache** in VRAM with LRU + cost-aware eviction. It exposes residency state to the optimizer.
- **Pinned host staging buffers** reused across queries.
- **Streaming mode:** when a pipeline's inputs exceed the budget, it processes chunk by chunk on 2–3 CUDA streams, overlapping H2D copy with compute (double/triple buffering). Partial aggregates merge at the end.

### 6.6 CPU backend
- Must be **genuinely good**, otherwise the optimizer's decisions mean nothing and the speedup numbers are inflated.
- Vectorized (batches of 2048 values), multithreaded with a work-stealing thread pool, compiler auto-vectorization + AVX2 where it helps.
- Implements the same operator set, so any operator can run on either device.

### 6.7 Placement optimizer (headline feature #2)
**Cost model per operator *o* on device *d*:**

```
cost(o, d) = compute(o, d, rows, width, selectivity, NDV)
           + Σ transfer(column c → d)   if c not resident on d
           − overlap(transfer, compute) if streaming possible
           + launch/sync overhead(d)
```

- **Calibration ("hardware fingerprint")** at first run: microbenchmarks measure PCIe H2D/D2H bandwidth (pinned and pageable), kernel launch latency, GPU/CPU throughput per primitive over a grid of sizes. The results are stored as a per-machine profile.
- **Learned refinement:** compute terms start as analytical formulas fitted to calibration data (piecewise-linear in rows). After each query, observed runtimes update the model by online least squares. This is the HyPE idea; our additions are the residency and overlap terms.
- **Search:** dynamic programming over the plan tree with state = (device of output, residency set). Plans are small, so this is exact.
- **Cardinality estimation:** zone maps + sampled histograms + NDV sketches (HyperLogLog) collected at load time.
- **Explainability:** every decision records alternatives and their predicted costs, and the dashboard shows them.
- **Evaluation:** compare `auto` against `always-CPU`, `always-GPU` and an **oracle** (brute-force all placements on small plans). Report hit rate and regret across data sizes from 1K to 100M rows, cold vs. warm cache.

### 6.8 Observability
- Every query emits a **trace**: plan, per-operator device/rows/predicted cost/actual time, kernel and memcpy events (CUDA events + optionally NVTX ranges for Nsight Systems).
- Exported as JSON so the dashboard and benchmark harness share one format.

## 7. Python layer

| Component | Description |
|---|---|
| `warpdb` Python module | pybind11 bindings: `connect()`, `execute(sql) → pyarrow/pandas`, `explain()`, `trace()` |
| Benchmark harness | Generates SSB/TPC-H data (`dbgen` or DuckDB's `tpch` extension), runs WarpDB-CPU, WarpDB-GPU, WarpDB-auto, DuckDB, SQLite, Polars. Writes CSV + plots |
| Differential fuzzer | Random query generator over the supported grammar, compares results with DuckDB (tolerance for floats, order-insensitive unless ORDER BY), auto-shrinks failing queries |
| **Dashboard** (Streamlit + Plotly) | SQL editor and results; plan DAG coloured by device; predicted-vs-actual cost bars; CUDA timeline (Gantt of kernels/copies per stream); VRAM usage over time; "what-if" slider to force placement and compare; benchmark leaderboard |

## 8. Success metrics (what goes on the résumé)

| Metric | Target |
|---|---|
| Correctness | 100% match with DuckDB on SSB + TPC-H subset; ≥10,000 fuzzed queries with zero mismatches |
| Primitive quality | Own scan, radix sort and compaction within 1.5× of CUB throughput |
| GPU vs own CPU | ≥10× on SSB Q1.x at SF10 (warm) |
| vs DuckDB | Report honestly; aim to beat DuckDB on warm, scan-heavy SSB queries |
| JIT fusion | Measurable speedup (expect 1.5–3×) over operator-at-a-time on multi-operator pipelines |
| Optimizer | ≥90% correct device choice, ≤10% mean regret vs oracle |
| Out-of-core | SF20+ completes on 8 GB VRAM via streaming, with an overlap efficiency figure |

## 9. Milestones

Estimates assume ~15–20 h/week. Each phase ends with something demoable.

| Phase | Weeks | Deliverable |
|---|---|---|
| **P0 Setup** | 0.5 | CMake/Ninja/MSVC (or WSL2) toolchain, repo skeleton, CI build, GoogleTest, a "hello kernel" on sm_120 |
| **P1 Storage + CPU engine** | 2 | Column store, CSV/SQLite ingest, parser, binder, CPU operators, results verified against DuckDB. *Demo: SQL runs on CPU* |
| **P2 GPU primitives** | 2.5 | Scan, compaction, radix sort, hash table, each with unit tests + microbenchmarks vs CUB/Thrust |
| **P3 GPU operators** | 2 | Filter, project, hash aggregation, hash join, sort, top-K. Memory manager + column cache. *Demo: SSB on GPU* |
| **P4 Optimizer v1** | 1.5 | Calibration, analytical cost model, residency-aware DP placement, `EXPLAIN` |
| **P5 Python + dashboard v1** | 1 | Bindings, benchmark harness, dashboard with plan + timeline |
| **P6 JIT fusion** | 2 | NVRTC codegen for pipelines, kernel cache, fused vs unfused benchmarks |
| **P7 Out-of-core + compression** | 1.5 | Chunked streaming with async overlap; dictionary + bit-packing |
| **P8 Learned costs + fuzzing** | 1 | Online model refinement, oracle evaluation, differential fuzzer |
| **P9 Polish** | 1 | README with results, architecture write-up (blog-post quality), demo video |

**Total ≈ 15 weeks.** P1–P5 alone (~9.5 weeks) is a strong, complete project. P6–P8 are what make it exceptional.

## 10. Repository layout (proposed)

```
warpdb/
  CMakeLists.txt
  src/
    common/       types, status, logging, thread pool
    storage/      column, chunk, catalog, encodings, ingest (csv, sqlite, wcol)
    sql/          lexer, parser, ast, binder
    plan/         logical plan, rewrites, physical plan
    optimizer/    cost model, calibration, placement DP, cardinality
    exec/cpu/     vectorized operators
    exec/gpu/     kernels/ (scan, compact, radix_sort, hash_table, join, agg, topk)
                  jit/ (codegen, nvrtc wrapper, cache)
                  memory/ (pool, column cache, staging)
    trace/        telemetry
    cli/          REPL
  python/         bindings, harness, fuzzer, dashboard
  tests/          unit (gtest), sql golden tests, fuzz
  bench/          microbenchmarks, SSB/TPC-H runners
  docs/
```

## 11. Testing strategy
- **Unit tests** per kernel with randomized sizes, including edge cases: 0, 1, not a multiple of block size, all-pass/no-pass predicates, duplicate keys, skewed keys (Zipf).
- **Golden SQL tests:** query + expected output files generated by DuckDB.
- **Device parity:** every SQL test runs under `device=cpu`, `gpu`, `auto`, and with fusion on/off.
- **Differential fuzzing** against DuckDB (§7).
- `compute-sanitizer` (memcheck, racecheck) runs on the kernel test suite.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Windows + CUDA + CMake friction | Decide native vs WSL2 in P0. Keep the build simple and do no Linux-only tricks |
| 8 GB VRAM limits dataset size | That is why streaming mode exists, and it turns the limit into a feature |
| Laptop GPU thermal throttling distorts benchmarks | Fixed warm-up, median of N runs, record clocks via NVML, plugged in |
| JIT complexity balloons | Fixed codegen templates per operator. Interpreted path always available |
| CPU baseline too weak → fake speedups | Compare against DuckDB too and report it honestly |
| Scope creep in SQL features | SQL scope (§5) is frozen for v1. Anything else is v2 |

## 13. Open questions (decide before P0)
1. **Native Windows (MSVC + CUDA) or WSL2 (gcc + CUDA)?** WSL2 makes Linux tooling and cudf-polars baselines easier. Native is simpler for the GPU profiler UI and has no virtualization quirks. *Recommendation: native Windows, with DuckDB/Polars as baselines.*
2. **Solo or team?** It changes the timeline and could split work as front end/CPU vs. GPU kernels vs. optimizer/dashboard.
3. **Deadline?** (placement season date). It sets the cut line between P5 and P9.
4. **Name** — "WarpDB" is a placeholder.

## 14. References
- Sirius / "Rethinking Analytical Processing in the GPU Era", CIDR 2026 — https://arxiv.org/abs/2508.04701 · https://vldb.org/cidrdb/2026/rethinking-analytical-processing-in-the-gpu-era.html
- NVIDIA cuDF for engine builders — https://perspectives.nvidia.com/cudf/building-data-engines-without-writing-custom-gpu-kernels/
- Velox/Presto GPU, VLDB 2026 ADMS — https://www.vldb.org/2026/Workshops/VLDB-Workshops-2026/ADMS/ADMS26_1.pdf
- HeavyDB status — https://dbdb.io/db/heavydb
- Breß et al., CoGaDB / HyPE — https://readingxtra.github.io/docs/db_GPU/BrebVLDB2017.pdf · https://hgpu.org/?p=12666
- Shanbhag, Madden, Yu, Crystal (SIGMOD 2020) — https://arxiv.org/abs/2003.01178 · https://github.com/kunliny/crystal
- Wu, Koutsoukos, Alonso, Joins & grouped aggregations on GPUs (SIGMOD 2025) — https://arxiv.org/html/2312.00720v2
- Merrill & Garland, Single-pass Parallel Prefix Scan with Decoupled Look-back (NVIDIA, 2016)

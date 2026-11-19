# JIT pipeline fusion - TPC-H SF1

GPU, columns resident in VRAM, median of 15 runs, client-side wall-clock ms. *Unfused* = operator-at-a-time kernels, every expression node materializes a vector; *fused* = NVRTC-compiled pipelines (filter+compact, and scan/filter/group/aggregate in one kernel). *NVRTC compile* is the one-off cost of the first run with an empty kernel cache; *new process* is the first fused run of a fresh process (columns already resident) that loads its kernels from the on-disk cache (no NVRTC).

| query | unfused ms | fused ms | speedup | NVRTC compile ms | first fused run, new process ms |
|---|---|---|---|---|---|
| q1 | 20.26 | 12.08 | 1.68x | 529 | 22.5 |
| q3 | 8.18 | 7.45 | 1.10x | 328 | 13.5 |
| q5 | 9.90 | 8.27 | 1.20x | 288 | 29.8 |
| q6 | 2.60 | 1.02 | 2.55x | 32 | 4.9 |
| q10 | 7.28 | 6.24 | 1.17x | 327 | 11.8 |
| q12 | 5.54 | 3.58 | 1.55x | 277 | 10.7 |
| q14 | 2.61 | 1.40 | 1.86x | 57 | 3.1 |
| count_star | 0.55 | 0.59 | 0.92x | 29 | 1.2 |
| between_dates | 1.66 | 0.97 | 1.72x | 32 | 1.9 |
| or_and | 2.15 | 1.09 | 1.98x | 31 | 1.7 |
| expr_arith | 6.39 | 2.03 | 3.14x | 39 | 2.6 |
| agg_in_arith | 7.59 | 5.07 | 1.49x | 266 | 8.4 |
| extract_group | 4.24 | 2.72 | 1.56x | 271 | 4.0 |
| count_expr | 1.83 | 1.04 | 1.76x | 36 | 4.2 |
| **geomean** | 4.06 | 2.53 | 1.61x | 182 (mean) | 5.5 |

# WarpDB GPU primitive benchmarks

GPU: NVIDIA GeForce RTX 5050 Laptop GPU (sm_120), 384 GB/s peak DRAM bandwidth, median of 15 runs

sizes: stream ops n=67108864, sort n=33554432, hash n=16777216

### Prefix scan (exclusive, uint32)

| impl | ms | GB/s | % of peak BW | vs CUB |
|---|---|---|---|---|
| WarpDB decoupled look-back | 2.128 | 252.3 | 66% | 0.99x |
| CUB DeviceScan | 2.111 | 254.3 | 66% | - |

### Stream compaction / filter (uint32, predicate v%100 < p)

| selectivity | impl | ms | GB/s (read+write) | vs CUB |
|---|---|---|---|---|
| 1% | WarpDB fused (ballot + look-back) | 1.175 | 230.8 | 1.12x |
| 1% | WarpDB 3-pass (flags, scan, scatter) | 6.896 | 39.3 | 0.19x |
| 1% | CUB DeviceSelect::If | 1.321 | 205.2 | - |
| 50% | WarpDB fused (ballot + look-back) | 1.674 | 240.6 | 0.98x |
| 50% | WarpDB 3-pass (flags, scan, scatter) | 6.908 | 58.3 | 0.23x |
| 50% | CUB DeviceSelect::If | 1.657 | 243.1 | - |
| 99% | WarpDB fused (ballot + look-back) | 2.159 | 247.5 | 0.98x |
| 99% | WarpDB 3-pass (flags, scan, scatter) | 6.884 | 77.6 | 0.30x |
| 99% | CUB DeviceSelect::If | 2.125 | 251.3 | - |

### Radix sort (random keys)

| keys | WarpDB ms | WarpDB Mkeys/s | CUB ms | CUB Mkeys/s | WarpDB vs CUB |
|---|---|---|---|---|---|
| uint32 | 7.39 | 4540 | 4.90 | 6850 | 0.66x |
| uint32 + uint32 row id | 10.85 | 3092 | 8.96 | 3744 | 0.83x |
| uint64 | 22.15 | 1515 | 18.01 | 1863 | 0.81x |
| uint64 + uint32 row id | 31.04 | 1081 | 26.68 | 1258 | 0.86x |
| int64 (small range) + row id | 30.20 | 1111 | 26.24 | 1279 | 0.87x |

### Hash table insert (int64 keys) vs sort-based distinct

| distinct keys | hash insert ms | M inserts/s | sort+RLE ms | speedup |
|---|---|---|---|---|
| 1000 | 0.95 | 17696 | 9.40 | 9.91x |
| 1048576 | 1.38 | 12168 | 9.43 | 6.84x |
| 16777216 | 11.08 | 1514 | 9.83 | 0.89x |

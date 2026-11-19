# TPC-H SF1 - WarpDB CPU vs GPU vs DuckDB

Median of 9 runs, milliseconds, client-side wall time including result transfer. `GPU cold` starts with an empty VRAM column cache (includes the PCIe upload of the needed columns); `GPU warm` has the columns already resident.

| query | WarpDB CPU | WarpDB GPU cold | WarpDB GPU warm | DuckDB | GPU warm vs CPU | GPU warm vs DuckDB |
|---|---|---|---|---|---|---|
| q1 | 398.2 | 237.8 | 27.2 | 49.6 | 14.7x | 1.83x |
| q3 | 165.5 | 55.6 | 8.8 | 84.8 | 18.8x | 9.65x |
| q5 | 201.4 | 88.2 | 9.0 | 58.9 | 22.4x | 6.55x |
| q6 | 178.5 | 83.0 | 3.1 | 13.3 | 58.0x | 4.32x |
| q10 | 185.8 | 150.6 | 8.1 | 205.7 | 22.9x | 25.30x |
| q12 | 181.3 | 134.3 | 6.4 | 58.5 | 28.2x | 9.11x |
| q14 | 111.6 | 41.0 | 2.6 | 33.4 | 42.1x | 12.59x |
| **geomean** | 189.9 | 96.9 | 7.1 | 53.9 | 26.7x | 7.59x |

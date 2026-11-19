# Placement optimizer evaluation

Oracle = faster of forced-CPU and forced-GPU for the same query in the same state. *warm* = columns already resident in VRAM, *cold* = VRAM column cache emptied before every run (GPU pays the PCIe upload). All times are client-side wall-clock milliseconds, median of several runs.

Percentage regret is inflated for sub-millisecond queries (the optimizer's own ~0.3 ms is a large fraction of a 0.4 ms query). The *high-stakes* column counts only decisions where the two devices differ by at least 1.5x and 1 ms, and the totals show what a whole workload actually costs.

## Summary

| scale | state | queries | hit rate | high-stakes hit rate | mean regret | auto total ms | always-CPU total | always-GPU total | oracle total |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | cold | 24 | 88% | 93% (14) | 31.9% | 38 | 52 | 57 | 35 |
| 0.01 | warm | 24 | 92% | 86% (14) | 30.3% | 34 | 53 | 49 | 31 |
| 0.1 | cold | 24 | 96% | 95% (22) | 11.0% | 122 | 581 | 133 | 125 |
| 0.1 | warm | 24 | 96% | 100% (21) | 24.6% | 56 | 653 | 61 | 51 |
| 1.0 | cold | 24 | 100% | 100% (23) | 8.9% | 391 | 2772 | 446 | 438 |
| 1.0 | warm | 24 | 100% | 100% (24) | 8.2% | 119 | 2612 | 156 | 152 |
| all | all | 144 | 95% | 97% (118) | 19.1% | 759 | 6724 | 902 | 832 |

## Per query

| SF | state | query | CPU ms | GPU ms | auto ms | auto chose | oracle | regret |
|---|---|---|---|---|---|---|---|---|
| 0.01 | warm | q1 | 13.83 | 3.63 | 3.87 | gpu | gpu | 7% |
| 0.01 | warm | q3 | 2.79 | 4.64 | 4.23 | gpu | cpu | 52% ❌ |
| 0.01 | warm | q5 | 3.28 | 4.08 | 3.47 | cpu | cpu | 6% |
| 0.01 | warm | q6 | 3.07 | 1.42 | 1.01 | gpu | gpu | 0% |
| 0.01 | warm | q10 | 2.91 | 5.16 | 2.80 | cpu | cpu | 0% |
| 0.01 | warm | q12 | 2.26 | 3.84 | 3.25 | gpu | cpu | 44% ❌ |
| 0.01 | warm | q14 | 1.64 | 1.69 | 1.85 | gpu | cpu | 13% |
| 0.01 | warm | count_star | 0.33 | 0.36 | 0.25 | gpu | cpu | 0% |
| 0.01 | warm | min_max_date | 1.98 | 1.31 | 0.64 | gpu | gpu | 0% |
| 0.01 | warm | distinct | 2.12 | 2.66 | 1.98 | cpu | cpu | 0% |
| 0.01 | warm | having | 2.56 | 2.92 | 3.35 | cpu | cpu | 31% |
| 0.01 | warm | extract_group | 0.77 | 1.93 | 0.82 | cpu | cpu | 6% |
| 0.01 | warm | order_alias_limit | 0.32 | 1.74 | 0.34 | cpu | cpu | 6% |
| 0.01 | warm | join_alias | 0.18 | 2.98 | 0.28 | cpu | cpu | 55% |
| 0.01 | warm | three_way | 0.13 | 2.99 | 0.15 | cpu | cpu | 10% |
| 0.01 | warm | multi_key_join | 5.07 | 1.19 | 0.53 | gpu | gpu | 0% |
| 0.01 | warm | agg_in_arith | 2.43 | 1.73 | 2.04 | gpu | gpu | 18% |
| 0.01 | warm | like_mid | 0.24 | 0.33 | 0.09 | cpu | cpu | 0% |
| 0.01 | warm | between_dates | 1.87 | 0.46 | 0.70 | gpu | gpu | 54% |
| 0.01 | warm | or_and | 2.08 | 0.74 | 0.67 | gpu | gpu | 0% |
| 0.01 | warm | in_list_int | 0.64 | 0.61 | 0.45 | cpu | gpu | 0% |
| 0.01 | warm | count_expr | 1.83 | 0.50 | 0.62 | gpu | gpu | 25% |
| 0.01 | warm | select_star_limit | 0.07 | 1.24 | 0.27 | cpu | cpu | 285% |
| 0.01 | warm | str_range | 0.11 | 0.72 | 0.24 | cpu | cpu | 118% |
| 0.01 | cold | q1 | 11.43 | 5.28 | 5.21 | gpu | gpu | 0% |
| 0.01 | cold | q3 | 2.28 | 5.67 | 2.58 | cpu | cpu | 14% |
| 0.01 | cold | q5 | 2.99 | 4.86 | 2.97 | cpu | cpu | 0% |
| 0.01 | cold | q6 | 2.49 | 1.67 | 1.76 | gpu | gpu | 6% |
| 0.01 | cold | q10 | 2.83 | 5.80 | 2.85 | cpu | cpu | 1% |
| 0.01 | cold | q12 | 2.27 | 3.52 | 3.27 | gpu | cpu | 44% ❌ |
| 0.01 | cold | q14 | 2.03 | 1.97 | 1.81 | cpu | gpu | 0% |
| 0.01 | cold | count_star | 0.35 | 0.47 | 0.97 | gpu | cpu | 173% ❌ |
| 0.01 | cold | min_max_date | 3.12 | 1.40 | 0.79 | gpu | gpu | 0% |
| 0.01 | cold | distinct | 2.36 | 2.98 | 2.17 | cpu | cpu | 0% |
| 0.01 | cold | having | 2.62 | 3.48 | 2.92 | cpu | cpu | 11% |
| 0.01 | cold | extract_group | 0.69 | 2.16 | 0.80 | cpu | cpu | 16% |
| 0.01 | cold | order_alias_limit | 0.23 | 2.43 | 0.49 | cpu | cpu | 112% |
| 0.01 | cold | join_alias | 0.21 | 2.91 | 0.19 | cpu | cpu | 0% |
| 0.01 | cold | three_way | 0.12 | 2.44 | 0.17 | cpu | cpu | 42% |
| 0.01 | cold | multi_key_join | 5.65 | 1.39 | 1.69 | gpu | gpu | 22% |
| 0.01 | cold | agg_in_arith | 2.92 | 2.57 | 3.12 | cpu | gpu | 22% ❌ |
| 0.01 | cold | like_mid | 0.13 | 0.67 | 0.11 | cpu | cpu | 0% |
| 0.01 | cold | between_dates | 2.26 | 1.33 | 0.84 | gpu | gpu | 0% |
| 0.01 | cold | or_and | 2.31 | 0.82 | 0.84 | gpu | gpu | 3% |
| 0.01 | cold | in_list_int | 0.54 | 0.81 | 0.47 | cpu | cpu | 0% |
| 0.01 | cold | count_expr | 1.97 | 0.95 | 1.78 | gpu | gpu | 88% |
| 0.01 | cold | select_star_limit | 0.08 | 1.12 | 0.24 | cpu | cpu | 182% |
| 0.01 | cold | str_range | 0.09 | 0.67 | 0.12 | cpu | cpu | 30% |
| 0.1 | warm | q1 | 106.06 | 5.98 | 5.80 | gpu | gpu | 0% |
| 0.1 | warm | q3 | 44.47 | 5.21 | 5.72 | gpu | gpu | 10% |
| 0.1 | warm | q5 | 78.61 | 5.19 | 5.05 | gpu | gpu | 0% |
| 0.1 | warm | q6 | 44.93 | 1.65 | 1.42 | gpu | gpu | 0% |
| 0.1 | warm | q10 | 32.18 | 4.39 | 4.26 | gpu | gpu | 0% |
| 0.1 | warm | q12 | 35.97 | 3.39 | 3.01 | gpu | gpu | 0% |
| 0.1 | warm | q14 | 26.88 | 1.60 | 1.24 | gpu | gpu | 0% |
| 0.1 | warm | count_star | 3.44 | 0.22 | 0.34 | gpu | gpu | 55% |
| 0.1 | warm | min_max_date | 20.22 | 0.78 | 1.24 | gpu | gpu | 59% |
| 0.1 | warm | distinct | 21.44 | 2.04 | 2.15 | gpu | gpu | 6% |
| 0.1 | warm | having | 20.01 | 2.95 | 3.05 | gpu | gpu | 3% |
| 0.1 | warm | extract_group | 8.55 | 2.24 | 5.39 | gpu | gpu | 141% |
| 0.1 | warm | order_alias_limit | 4.71 | 5.87 | 2.37 | gpu | cpu | 0% |
| 0.1 | warm | join_alias | 0.45 | 3.36 | 0.50 | cpu | cpu | 10% |
| 0.1 | warm | three_way | 0.35 | 2.73 | 0.45 | cpu | cpu | 28% |
| 0.1 | warm | multi_key_join | 63.46 | 1.48 | 1.51 | gpu | gpu | 2% |
| 0.1 | warm | agg_in_arith | 26.10 | 3.03 | 6.22 | gpu | gpu | 105% |
| 0.1 | warm | like_mid | 0.51 | 0.68 | 0.82 | gpu | cpu | 63% |
| 0.1 | warm | between_dates | 34.69 | 0.91 | 0.78 | gpu | gpu | 0% |
| 0.1 | warm | or_and | 33.48 | 1.08 | 0.97 | gpu | gpu | 0% |
| 0.1 | warm | in_list_int | 15.31 | 1.07 | 0.98 | gpu | gpu | 0% |
| 0.1 | warm | count_expr | 30.23 | 0.67 | 0.69 | gpu | gpu | 2% |
| 0.1 | warm | select_star_limit | 0.28 | 3.04 | 0.43 | cpu | cpu | 52% |
| 0.1 | warm | str_range | 0.75 | 1.55 | 1.15 | gpu | cpu | 54% ❌ |
| 0.1 | cold | q1 | 87.85 | 19.23 | 22.41 | gpu | gpu | 17% |
| 0.1 | cold | q3 | 34.39 | 7.88 | 11.60 | gpu | gpu | 47% |
| 0.1 | cold | q5 | 63.31 | 11.45 | 12.70 | gpu | gpu | 11% |
| 0.1 | cold | q6 | 42.21 | 10.82 | 9.08 | gpu | gpu | 0% |
| 0.1 | cold | q10 | 41.42 | 14.28 | 11.34 | gpu | gpu | 0% |
| 0.1 | cold | q12 | 32.46 | 8.83 | 7.49 | gpu | gpu | 0% |
| 0.1 | cold | q14 | 27.70 | 7.48 | 6.65 | gpu | gpu | 0% |
| 0.1 | cold | count_star | 4.89 | 0.27 | 0.29 | gpu | gpu | 7% |
| 0.1 | cold | min_max_date | 21.26 | 2.20 | 2.04 | gpu | gpu | 0% |
| 0.1 | cold | distinct | 15.42 | 3.22 | 2.44 | gpu | gpu | 0% |
| 0.1 | cold | having | 18.09 | 5.08 | 5.14 | gpu | gpu | 1% |
| 0.1 | cold | extract_group | 8.79 | 3.38 | 2.66 | gpu | gpu | 0% |
| 0.1 | cold | order_alias_limit | 3.68 | 3.66 | 2.05 | gpu | gpu | 0% |
| 0.1 | cold | join_alias | 0.30 | 2.95 | 0.23 | cpu | cpu | 0% |
| 0.1 | cold | three_way | 0.15 | 2.85 | 0.18 | cpu | cpu | 24% |
| 0.1 | cold | multi_key_join | 47.72 | 4.28 | 4.18 | gpu | gpu | 0% |
| 0.1 | cold | agg_in_arith | 23.38 | 7.60 | 7.18 | gpu | gpu | 0% |
| 0.1 | cold | like_mid | 0.46 | 0.62 | 0.44 | cpu | cpu | 0% |
| 0.1 | cold | between_dates | 31.30 | 2.51 | 2.57 | gpu | gpu | 2% |
| 0.1 | cold | or_and | 33.23 | 4.52 | 4.03 | gpu | gpu | 0% |
| 0.1 | cold | in_list_int | 15.29 | 1.31 | 1.41 | gpu | gpu | 8% |
| 0.1 | cold | count_expr | 27.20 | 5.12 | 4.48 | gpu | gpu | 0% |
| 0.1 | cold | select_star_limit | 0.26 | 1.77 | 0.12 | cpu | cpu | 0% |
| 0.1 | cold | str_range | 0.41 | 1.47 | 1.01 | gpu | cpu | 145% ❌ |
| 1 | warm | q1 | 337.86 | 28.85 | 26.64 | gpu | gpu | 0% |
| 1 | warm | q3 | 176.81 | 7.18 | 7.52 | gpu | gpu | 5% |
| 1 | warm | q5 | 214.07 | 9.33 | 9.63 | gpu | gpu | 3% |
| 1 | warm | q6 | 148.31 | 4.61 | 2.90 | gpu | gpu | 0% |
| 1 | warm | q10 | 201.31 | 8.14 | 6.85 | gpu | gpu | 0% |
| 1 | warm | q12 | 165.19 | 31.24 | 5.60 | gpu | gpu | 0% |
| 1 | warm | q14 | 115.03 | 3.34 | 3.30 | gpu | gpu | 0% |
| 1 | warm | count_star | 21.90 | 1.10 | 1.35 | gpu | gpu | 22% |
| 1 | warm | min_max_date | 92.83 | 4.50 | 4.20 | gpu | gpu | 0% |
| 1 | warm | distinct | 87.61 | 4.62 | 3.86 | gpu | gpu | 0% |
| 1 | warm | having | 192.88 | 6.32 | 5.82 | gpu | gpu | 0% |
| 1 | warm | extract_group | 45.01 | 3.83 | 3.96 | gpu | gpu | 3% |
| 1 | warm | order_alias_limit | 33.16 | 5.05 | 4.80 | gpu | gpu | 0% |
| 1 | warm | join_alias | 1.04 | 3.24 | 0.76 | cpu | cpu | 0% |
| 1 | warm | three_way | 0.63 | 2.11 | 0.73 | cpu | cpu | 15% |
| 1 | warm | multi_key_join | 330.85 | 8.55 | 8.74 | gpu | gpu | 2% |
| 1 | warm | agg_in_arith | 100.72 | 11.74 | 11.29 | gpu | gpu | 0% |
| 1 | warm | like_mid | 7.60 | 0.92 | 1.01 | gpu | gpu | 10% |
| 1 | warm | between_dates | 79.47 | 2.31 | 2.01 | gpu | gpu | 0% |
| 1 | warm | or_and | 113.49 | 2.99 | 2.39 | gpu | gpu | 0% |
| 1 | warm | in_list_int | 57.53 | 0.83 | 0.91 | gpu | gpu | 10% |
| 1 | warm | count_expr | 79.52 | 3.15 | 2.54 | gpu | gpu | 0% |
| 1 | warm | select_star_limit | 0.31 | 1.40 | 0.20 | cpu | cpu | 0% |
| 1 | warm | str_range | 9.24 | 0.99 | 2.25 | gpu | gpu | 126% |
| 1 | cold | q1 | 385.20 | 67.42 | 64.27 | gpu | gpu | 0% |
| 1 | cold | q3 | 162.82 | 39.06 | 33.28 | gpu | gpu | 0% |
| 1 | cold | q5 | 203.46 | 33.65 | 29.50 | gpu | gpu | 0% |
| 1 | cold | q6 | 148.20 | 25.74 | 25.57 | gpu | gpu | 0% |
| 1 | cold | q10 | 235.97 | 31.08 | 31.42 | gpu | gpu | 1% |
| 1 | cold | q12 | 193.66 | 23.06 | 29.79 | gpu | gpu | 29% |
| 1 | cold | q14 | 116.51 | 19.47 | 20.47 | gpu | gpu | 5% |
| 1 | cold | count_star | 23.51 | 1.66 | 1.62 | gpu | gpu | 0% |
| 1 | cold | min_max_date | 150.21 | 14.53 | 15.79 | gpu | gpu | 9% |
| 1 | cold | distinct | 95.53 | 11.09 | 10.75 | gpu | gpu | 0% |
| 1 | cold | having | 178.62 | 53.02 | 20.03 | gpu | gpu | 0% |
| 1 | cold | extract_group | 46.86 | 12.15 | 17.62 | gpu | gpu | 45% |
| 1 | cold | order_alias_limit | 36.16 | 12.28 | 7.28 | gpu | gpu | 0% |
| 1 | cold | join_alias | 0.99 | 5.14 | 0.89 | cpu | cpu | 0% |
| 1 | cold | three_way | 1.03 | 3.48 | 1.06 | cpu | cpu | 4% |
| 1 | cold | multi_key_join | 321.58 | 21.63 | 20.14 | gpu | gpu | 0% |
| 1 | cold | agg_in_arith | 96.21 | 25.75 | 24.46 | gpu | gpu | 0% |
| 1 | cold | like_mid | 3.89 | 0.68 | 1.16 | gpu | gpu | 71% |
| 1 | cold | between_dates | 103.51 | 9.87 | 10.85 | gpu | gpu | 10% |
| 1 | cold | or_and | 118.26 | 21.10 | 13.05 | gpu | gpu | 0% |
| 1 | cold | in_list_int | 58.03 | 2.87 | 3.04 | gpu | gpu | 6% |
| 1 | cold | count_expr | 86.24 | 8.95 | 7.34 | gpu | gpu | 0% |
| 1 | cold | select_star_limit | 0.09 | 0.94 | 0.12 | cpu | cpu | 34% |
| 1 | cold | str_range | 5.87 | 1.10 | 1.10 | gpu | gpu | 0% |

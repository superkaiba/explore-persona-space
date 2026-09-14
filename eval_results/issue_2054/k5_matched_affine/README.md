# Matched K5 representations: shift versus shift plus scalar

This extends the direct paired-representation test, not the earlier context-to-answer map calibration. Each context vector predicts the corresponding context vector in another setting; answer vectors predict answer vectors. The strict parent query cohorts and global conversation folds are unchanged.

Scaling reduces held-out squared error in 120/120 directed panels, while lowering mean Euclidean top-1 retrieval in 120/120. This is a predictive error reduction with a query-retrieval tradeoff, not evidence that every setting is a reversibly scaled and shifted copy of another.

y_hat = a*x+b, with one scalar a across all coordinates and paired training queries, plus an unconstrained vector b. Both fitted separately for each direction, arm, model, and held-out fold. Comparators: identity, identity plus paired-training mean difference.

1 - sum_heldout ||y-(a*x+b)||^2 / sum_heldout ||y-x||^2; pooled over five folds. 100% is perfect paired reconstruction; 0% matches identity. This is not centered R2 and scaling need not give symmetric scores.

For training matrices X and Y, a = sum((X−mean X)*(Y−mean Y)) / sum((X−mean X)^2), and b = mean Y − a*mean X. There is one shared scalar per direction, model, representation arm and fold, not a separate scale per query or dimension. No regularization or hyperparameter search is needed. Source and target training pairs are used; no held-out target rows enter fitting.

Both directions are fitted independently: noisy forward and reverse scales generally are not reciprocals. Shrinkage can improve prediction without providing a reversible transformation. Centered R² and nearest-neighbor retrieval therefore accompany the identity-relative displacement fraction.

Coverage: 120 directed representation panels and 600 fold evaluations. Every shift-only per-query error was checked against the previous strict result; no cohort or baseline drift was detected. Pool size is the held-out paired fold, with chance top-1 retrieval 1/pool size.

## Group summaries

Values below are unweighted means across the indicated directed setting pairs, after pooling displacement errors across held-out queries within each pair. Parent pair cohorts differ; the JSON includes per-pair and fold values and ranges.

| Model | Arm | Direction | Difference explained: shift → scale | R²: shift → scale | Top-1: shift → scale | Scalar range across folds |
|---|---|---|---:|---:|---:|---:|
| qwen2.5-7b | context | Chat → plain | 59.79% → 75.61% | -0.5863 → 0.0378 | 10.90% → 0.36% | 0.198–0.200 |
| qwen2.5-7b | context | Plain → chat | 59.79% → 76.27% | -0.6304 → 0.0380 | 14.91% → 0.33% | 0.193–0.195 |
| qwen2.5-7b | context | Chat → characters | 67.02% → 80.39% | -0.6503 → 0.0190 | 11.05% → 0.43% | 0.136–0.166 |
| qwen2.5-7b | context | Characters → chat | 67.02% → 81.79% | -0.7764 → 0.0196 | 13.13% → 0.46% | 0.127–0.156 |
| qwen2.5-7b | context | Plain → characters | 66.17% → 79.62% | -0.6139 → 0.0280 | 18.71% → 0.73% | 0.159–0.196 |
| qwen2.5-7b | context | Characters → plain | 66.17% → 80.65% | -0.7010 → 0.0281 | 14.78% → 0.55% | 0.152–0.189 |
| qwen2.5-7b | context | Characters → characters | 17.74% → 41.52% | -0.1633 → 0.1737 | 53.52% → 25.79% | 0.385–0.465 |
| qwen2.5-7b | answer | Chat → plain | 11.51% → 28.36% | 0.0605 → 0.2394 | 29.27% → 11.67% | 0.533–0.545 |
| qwen2.5-7b | answer | Plain → chat | 11.51% → 40.29% | -0.1271 → 0.2398 | 33.96% → 11.19% | 0.446–0.448 |
| qwen2.5-7b | answer | Chat → characters | 38.63% → 50.05% | -0.0761 → 0.1304 | 26.95% → 6.73% | 0.355–0.511 |
| qwen2.5-7b | answer | Characters → chat | 38.63% → 67.37% | -0.6416 → 0.1306 | 33.34% → 3.96% | 0.264–0.321 |
| qwen2.5-7b | answer | Plain → characters | 37.79% → 51.67% | -0.1013 → 0.1542 | 33.16% → 9.33% | 0.341–0.513 |
| qwen2.5-7b | answer | Characters → plain | 37.79% → 62.06% | -0.3876 → 0.1545 | 37.35% → 5.89% | 0.324–0.386 |
| qwen2.5-7b | answer | Characters → characters | 10.17% → 30.84% | 0.0748 → 0.2978 | 62.49% → 38.96% | 0.433–0.645 |
| qwen2.5-7b-instruct | context | Chat → plain | 53.72% → 64.70% | 0.0832 → 0.3009 | 69.21% → 38.90% | 0.539–0.542 |
| qwen2.5-7b-instruct | context | Plain → chat | 53.72% → 63.61% | 0.1111 → 0.3010 | 82.20% → 44.50% | 0.557–0.558 |
| qwen2.5-7b-instruct | context | Chat → characters | 60.42% → 76.45% | -0.5632 → 0.0707 | 40.20% → 2.32% | 0.211–0.284 |
| qwen2.5-7b-instruct | context | Characters → chat | 60.42% → 73.35% | -0.3803 → 0.0711 | 32.57% → 2.05% | 0.239–0.324 |
| qwen2.5-7b-instruct | context | Plain → characters | 59.71% → 76.89% | -0.6880 → 0.0319 | 22.87% → 0.78% | 0.151–0.199 |
| qwen2.5-7b-instruct | context | Characters → plain | 59.71% → 75.37% | -0.5843 → 0.0322 | 14.90% → 0.43% | 0.160–0.219 |
| qwen2.5-7b-instruct | context | Characters → characters | 18.62% → 43.21% | -0.2158 → 0.1523 | 52.76% → 20.97% | 0.360–0.444 |
| qwen2.5-7b-instruct | answer | Chat → plain | 29.53% → 65.35% | -0.0447 → 0.4863 | 73.13% → 57.73% | 0.487–0.493 |
| qwen2.5-7b-instruct | answer | Plain → chat | 29.53% → 29.53% | 0.4865 → 0.4865 | 65.76% → 65.58% | 0.994–0.997 |
| qwen2.5-7b-instruct | answer | Chat → characters | 36.97% → 55.76% | -0.0530 → 0.2770 | 59.20% → 31.56% | 0.368–0.597 |
| qwen2.5-7b-instruct | answer | Characters → chat | 36.97% → 48.43% | 0.1163 → 0.2773 | 59.57% → 36.66% | 0.521–0.624 |
| qwen2.5-7b-instruct | answer | Plain → characters | 48.32% → 53.83% | 0.0826 → 0.1862 | 34.73% → 15.02% | 0.454–0.698 |
| qwen2.5-7b-instruct | answer | Characters → plain | 48.32% → 74.67% | -0.6602 → 0.1866 | 49.35% → 11.71% | 0.294–0.346 |
| qwen2.5-7b-instruct | answer | Characters → characters | 11.46% → 29.81% | 0.1708 → 0.3517 | 67.19% → 51.13% | 0.474–0.688 |

## Per-pair results

R² and retrieval entries list identity / shift / shift+scale, in that order.

| Model | Arm | Direction | N | Difference explained: shift → scale | R² | Top-1 (%) | Mean scalar | Pool range |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-7b | context | Chat → Plain | 7999 | 59.79% → 75.61% | -2.9461 / -0.5863 / 0.0378 | 3.55 / 10.90 / 0.36 | 0.199 | 1560–1656 |
| qwen2.5-7b | context | Plain → Chat | 7999 | 59.79% → 76.27% | -3.0555 / -0.6304 / 0.0380 | 5.99 / 14.91 / 0.33 | 0.194 | 1560–1656 |
| qwen2.5-7b | answer | Chat → Plain | 7999 | 11.51% → 28.36% | -0.0617 / 0.0605 / 0.2394 | 25.67 / 29.27 / 11.67 | 0.537 | 1560–1656 |
| qwen2.5-7b | answer | Plain → Chat | 7999 | 11.51% → 40.29% | -0.2738 / -0.1271 / 0.2398 | 32.65 / 33.96 / 11.19 | 0.447 | 1560–1656 |
| qwen2.5-7b | context | Chat → HELIOS | 3044 | 67.32% → 80.67% | -4.0557 / -0.6519 / 0.0229 | 2.38 / 13.30 / 0.53 | 0.162 | 572–648 |
| qwen2.5-7b | context | HELIOS → Chat | 3044 | 67.32% → 81.47% | -4.2735 / -0.7231 / 0.0231 | 5.81 / 14.62 / 0.71 | 0.155 | 572–648 |
| qwen2.5-7b | answer | Chat → HELIOS | 3044 | 37.02% → 47.27% | -0.5998 / -0.0077 / 0.1563 | 17.28 / 29.33 / 8.94 | 0.496 | 572–648 |
| qwen2.5-7b | answer | HELIOS → Chat | 3044 | 37.02% → 65.97% | -1.4831 / -0.5624 / 0.1574 | 28.18 / 37.58 / 5.95 | 0.320 | 572–648 |
| qwen2.5-7b | context | Chat → Wren | 2763 | 66.00% → 79.90% | -3.8895 / -0.6624 / 0.0172 | 1.60 / 10.41 / 0.29 | 0.146 | 512–565 |
| qwen2.5-7b | context | Wren → Chat | 2763 | 66.00% → 81.33% | -4.2642 / -0.7897 / 0.0176 | 4.54 / 11.77 / 0.32 | 0.135 | 512–565 |
| qwen2.5-7b | answer | Chat → Wren | 2763 | 36.57% → 48.37% | -0.6743 / -0.0617 / 0.1354 | 15.43 / 28.14 / 6.85 | 0.455 | 512–565 |
| qwen2.5-7b | answer | Wren → Chat | 2763 | 36.57% → 65.70% | -1.5332 / -0.6044 / 0.1358 | 24.36 / 34.26 / 3.86 | 0.302 | 512–565 |
| qwen2.5-7b | context | Chat → Dana | 2640 | 67.63% → 80.23% | -3.9509 / -0.6025 / 0.0214 | 1.51 / 11.10 / 0.49 | 0.164 | 512–541 |
| qwen2.5-7b | context | Dana → Chat | 2640 | 67.63% → 82.39% | -4.5553 / -0.7978 / 0.0220 | 6.07 / 15.84 / 0.46 | 0.147 | 512–541 |
| qwen2.5-7b | answer | Chat → Dana | 2640 | 36.92% → 45.79% | -0.6049 / -0.0124 / 0.1298 | 13.17 / 24.87 / 7.11 | 0.491 | 512–541 |
| qwen2.5-7b | answer | Dana → Chat | 2640 | 36.92% → 70.24% | -1.9237 / -0.8443 / 0.1299 | 24.26 / 34.45 / 3.68 | 0.270 | 512–541 |
| qwen2.5-7b | context | Chat → Vex | 2546 | 67.13% → 80.77% | -4.1261 / -0.6842 / 0.0146 | 1.23 / 9.41 / 0.40 | 0.138 | 482–554 |
| qwen2.5-7b | context | Vex → Chat | 2546 | 67.13% → 81.97% | -4.4633 / -0.7950 / 0.0156 | 4.55 / 10.31 / 0.35 | 0.129 | 482–554 |
| qwen2.5-7b | answer | Chat → Vex | 2546 | 44.01% → 58.78% | -1.1835 / -0.2224 / 0.0999 | 11.41 / 25.47 / 4.01 | 0.360 | 482–554 |
| qwen2.5-7b | answer | Vex → Chat | 2546 | 44.01% → 67.57% | -1.7785 / -0.5554 / 0.0991 | 15.53 / 27.06 / 2.34 | 0.283 | 482–554 |
| qwen2.5-7b | context | Plain → HELIOS | 3044 | 66.61% → 80.06% | -3.8509 / -0.6198 / 0.0331 | 5.84 / 20.60 / 0.91 | 0.189 | 572–648 |
| qwen2.5-7b | context | HELIOS → Plain | 3044 | 66.61% → 80.29% | -3.9065 / -0.6380 / 0.0326 | 4.90 / 15.13 / 0.67 | 0.187 | 572–648 |
| qwen2.5-7b | answer | Plain → HELIOS | 3044 | 36.56% → 48.26% | -0.5606 / 0.0099 / 0.1922 | 22.92 / 36.63 / 13.29 | 0.508 | 572–648 |
| qwen2.5-7b | answer | HELIOS → Plain | 3044 | 36.56% → 61.04% | -1.0738 / -0.3158 / 0.1927 | 30.18 / 42.98 / 9.29 | 0.383 | 572–648 |
| qwen2.5-7b | context | Plain → Wren | 2763 | 64.64% → 78.93% | -3.6227 / -0.6342 / 0.0261 | 4.93 / 18.58 / 0.65 | 0.173 | 512–565 |
| qwen2.5-7b | context | Wren → Plain | 2763 | 64.64% → 79.73% | -3.8088 / -0.6999 / 0.0261 | 4.24 / 12.92 / 0.54 | 0.166 | 512–565 |
| qwen2.5-7b | answer | Plain → Wren | 2763 | 35.68% → 49.72% | -0.6767 / -0.0783 / 0.1570 | 20.91 / 34.28 / 9.35 | 0.451 | 512–565 |
| qwen2.5-7b | answer | Wren → Plain | 2763 | 35.68% → 60.76% | -1.1503 / -0.3827 / 0.1570 | 24.85 / 38.54 / 5.80 | 0.352 | 512–565 |
| qwen2.5-7b | context | Plain → Dana | 2640 | 67.10% → 79.46% | -3.7160 / -0.5519 / 0.0313 | 5.60 / 19.48 / 0.84 | 0.195 | 512–541 |
| qwen2.5-7b | context | Dana → Plain | 2640 | 67.10% → 81.59% | -4.2634 / -0.7314 / 0.0315 | 5.83 / 18.13 / 0.57 | 0.175 | 512–541 |
| qwen2.5-7b | answer | Plain → Dana | 2640 | 35.77% → 47.36% | -0.6086 / -0.0334 / 0.1532 | 18.54 / 31.17 / 9.69 | 0.478 | 512–541 |
| qwen2.5-7b | answer | Dana → Plain | 2640 | 35.77% → 64.08% | -1.3564 / -0.5136 / 0.1537 | 25.46 / 39.07 / 5.46 | 0.326 | 512–541 |
| qwen2.5-7b | context | Plain → Vex | 2546 | 66.33% → 80.03% | -3.9022 / -0.6495 / 0.0215 | 4.26 / 16.19 / 0.52 | 0.161 | 482–554 |
| qwen2.5-7b | context | Vex → Plain | 2546 | 66.33% → 81.01% | -4.1567 / -0.7346 / 0.0220 | 4.35 / 12.94 / 0.44 | 0.153 | 482–554 |
| qwen2.5-7b | answer | Plain → Vex | 2546 | 43.13% → 61.35% | -1.2915 / -0.3033 / 0.1143 | 14.40 / 30.54 / 5.00 | 0.346 | 482–554 |
| qwen2.5-7b | answer | Vex → Plain | 2546 | 43.13% → 62.37% | -1.3533 / -0.3381 / 0.1145 | 14.48 / 28.82 / 3.02 | 0.337 | 482–554 |
| qwen2.5-7b | context | HELIOS → Wren | 1431 | 22.64% → 44.49% | -0.4982 / -0.1593 / 0.1680 | 46.27 / 53.24 / 24.87 | 0.421 | 271–312 |
| qwen2.5-7b | context | Wren → HELIOS | 1431 | 22.64% → 46.06% | -0.5416 / -0.1929 / 0.1679 | 38.10 / 52.48 / 24.12 | 0.409 | 271–312 |
| qwen2.5-7b | answer | HELIOS → Wren | 1431 | 9.15% → 29.85% | 0.0447 / 0.1321 / 0.3296 | 62.47 / 66.74 / 46.78 | 0.565 | 271–312 |
| qwen2.5-7b | answer | Wren → HELIOS | 1431 | 9.15% → 26.85% | 0.0829 / 0.1668 / 0.3292 | 63.11 / 66.46 / 46.56 | 0.589 | 271–312 |
| qwen2.5-7b | context | HELIOS → Dana | 1390 | 28.38% → 48.16% | -0.5760 / -0.1283 / 0.1831 | 46.12 / 58.15 / 27.97 | 0.437 | 243–306 |
| qwen2.5-7b | context | Dana → HELIOS | 1390 | 28.38% → 49.17% | -0.6066 / -0.1502 / 0.1836 | 41.88 / 61.44 / 28.62 | 0.429 | 243–306 |
| qwen2.5-7b | answer | HELIOS → Dana | 1390 | 13.64% → 29.10% | 0.0544 / 0.1833 / 0.3294 | 58.08 / 66.55 / 46.87 | 0.602 | 243–306 |
| qwen2.5-7b | answer | Dana → HELIOS | 1390 | 13.64% → 34.85% | -0.0302 / 0.1105 / 0.3291 | 63.61 / 67.63 / 44.65 | 0.553 | 243–306 |
| qwen2.5-7b | context | HELIOS → Vex | 1313 | 22.05% → 46.03% | -0.5667 / -0.2208 / 0.1548 | 45.88 / 52.76 / 21.87 | 0.395 | 239–282 |
| qwen2.5-7b | context | Vex → HELIOS | 1313 | 22.05% → 44.87% | -0.5310 / -0.1931 / 0.1557 | 35.59 / 51.26 / 18.76 | 0.404 | 239–282 |
| qwen2.5-7b | answer | HELIOS → Vex | 1313 | 13.76% → 43.77% | -0.3143 / -0.1335 / 0.2621 | 53.95 / 62.70 / 32.60 | 0.451 | 239–282 |
| qwen2.5-7b | answer | Vex → HELIOS | 1313 | 13.76% → 26.74% | -0.0072 / 0.1314 / 0.2615 | 44.72 / 55.65 / 30.24 | 0.588 | 239–282 |
| qwen2.5-7b | context | Wren → Dana | 1337 | 7.89% → 32.60% | -0.1739 / -0.0814 / 0.2087 | 52.69 / 56.04 / 33.98 | 0.462 | 259–279 |
| qwen2.5-7b | context | Dana → Wren | 1337 | 7.89% → 32.83% | -0.1776 / -0.0849 / 0.2089 | 55.89 / 57.90 / 39.27 | 0.460 | 259–279 |
| qwen2.5-7b | answer | Wren → Dana | 1337 | 4.33% → 18.94% | 0.1956 / 0.2305 / 0.3479 | 63.58 / 67.10 / 49.25 | 0.634 | 259–279 |
| qwen2.5-7b | answer | Dana → Wren | 1337 | 4.33% → 29.10% | 0.0810 / 0.1208 / 0.3486 | 67.58 / 67.67 / 47.46 | 0.555 | 259–279 |
| qwen2.5-7b | context | Wren → Vex | 1282 | 10.42% → 38.32% | -0.3724 / -0.2292 / 0.1535 | 43.06 / 46.98 / 18.60 | 0.392 | 228–285 |
| qwen2.5-7b | context | Vex → Wren | 1282 | 10.42% → 36.52% | -0.3344 / -0.1952 / 0.1529 | 41.87 / 46.13 / 19.56 | 0.403 | 228–285 |
| qwen2.5-7b | answer | Wren → Vex | 1282 | 8.94% → 39.66% | -0.2336 / -0.1233 / 0.2550 | 56.34 / 59.94 / 31.19 | 0.453 | 228–285 |
| qwen2.5-7b | answer | Vex → Wren | 1282 | 8.94% → 23.88% | 0.0200 / 0.1077 / 0.2545 | 47.98 / 54.01 / 29.58 | 0.571 | 228–285 |
| qwen2.5-7b | context | Dana → Vex | 1304 | 15.06% → 40.63% | -0.3941 / -0.1842 / 0.1725 | 46.15 / 53.36 / 28.06 | 0.415 | 239–286 |
| qwen2.5-7b | context | Vex → Dana | 1304 | 15.06% → 38.54% | -0.3428 / -0.1406 / 0.1743 | 46.80 / 52.48 / 23.85 | 0.430 | 239–286 |
| qwen2.5-7b | answer | Dana → Vex | 1304 | 11.23% → 44.72% | -0.3329 / -0.1829 / 0.2633 | 58.04 / 62.79 / 31.72 | 0.437 | 239–286 |
| qwen2.5-7b | answer | Vex → Dana | 1304 | 11.23% → 22.60% | 0.0475 / 0.1546 / 0.2628 | 45.63 / 52.67 / 30.65 | 0.611 | 239–286 |
| qwen2.5-7b-instruct | context | Chat → Plain | 8000 | 53.72% → 64.70% | -0.9812 / 0.0832 / 0.3009 | 34.57 / 69.21 / 38.90 | 0.541 | 1560–1656 |
| qwen2.5-7b-instruct | context | Plain → Chat | 8000 | 53.72% → 63.61% | -0.9208 / 0.1111 / 0.3010 | 66.09 / 82.20 / 44.50 | 0.558 | 1560–1656 |
| qwen2.5-7b-instruct | answer | Chat → Plain | 8000 | 29.53% → 65.35% | -0.4826 / -0.0447 / 0.4863 | 54.65 / 73.13 / 57.73 | 0.489 | 1560–1656 |
| qwen2.5-7b-instruct | answer | Plain → Chat | 8000 | 29.53% → 29.53% | 0.2712 / 0.4865 / 0.4865 | 53.40 / 65.76 / 65.58 | 0.995 | 1560–1656 |
| qwen2.5-7b-instruct | context | Chat → HELIOS | 3047 | 60.88% → 76.19% | -2.8269 / -0.4961 / 0.0892 | 11.18 / 44.44 / 2.83 | 0.283 | 573–648 |
| qwen2.5-7b-instruct | context | HELIOS → Chat | 3047 | 60.88% → 72.94% | -2.3679 / -0.3167 / 0.0889 | 21.71 / 37.63 / 3.14 | 0.322 | 573–648 |
| qwen2.5-7b-instruct | answer | Chat → HELIOS | 3047 | 36.09% → 49.93% | -0.2658 / 0.1912 / 0.3661 | 49.68 / 66.91 / 50.95 | 0.592 | 573–648 |
| qwen2.5-7b-instruct | answer | HELIOS → Chat | 3047 | 36.09% → 47.45% | -0.2057 / 0.2296 / 0.3665 | 65.87 / 71.78 / 54.60 | 0.621 | 573–648 |
| qwen2.5-7b-instruct | context | Chat → Wren | 2765 | 58.93% → 75.70% | -2.8336 / -0.5744 / 0.0686 | 13.56 / 40.15 / 2.73 | 0.250 | 512–566 |
| qwen2.5-7b-instruct | context | Wren → Chat | 2765 | 58.93% → 72.32% | -2.3641 / -0.3815 / 0.0690 | 17.05 / 31.16 / 1.43 | 0.285 | 512–566 |
| qwen2.5-7b-instruct | answer | Chat → Wren | 2765 | 35.53% → 54.55% | -0.5542 / -0.0017 / 0.2942 | 44.33 / 61.12 / 37.09 | 0.501 | 512–566 |
| qwen2.5-7b-instruct | answer | Wren → Chat | 2765 | 35.53% → 46.28% | -0.3135 / 0.1532 / 0.2944 | 53.88 / 62.25 / 39.72 | 0.592 | 512–566 |
| qwen2.5-7b-instruct | context | Chat → Dana | 2641 | 61.25% → 76.51% | -2.9329 / -0.5239 / 0.0763 | 12.19 / 41.75 / 2.55 | 0.266 | 512–542 |
| qwen2.5-7b-instruct | context | Dana → Chat | 2641 | 61.25% → 73.88% | -2.5360 / -0.3700 / 0.0766 | 17.79 / 37.15 / 2.62 | 0.296 | 512–542 |
| qwen2.5-7b-instruct | answer | Chat → Dana | 2641 | 36.53% → 53.96% | -0.6203 / -0.0285 / 0.2539 | 37.59 / 56.65 / 25.79 | 0.488 | 512–542 |
| qwen2.5-7b-instruct | answer | Dana → Chat | 2641 | 36.53% → 50.50% | -0.5067 / 0.0437 / 0.2542 | 49.88 / 58.05 / 33.78 | 0.525 | 512–542 |
| qwen2.5-7b-instruct | context | Chat → Vex | 2547 | 60.63% → 77.41% | -3.2115 / -0.6583 / 0.0488 | 6.71 / 34.47 / 1.19 | 0.214 | 482–554 |
| qwen2.5-7b-instruct | context | Vex → Chat | 2547 | 60.63% → 74.26% | -2.6903 / -0.4531 / 0.0498 | 12.31 / 24.32 / 1.02 | 0.244 | 482–554 |
| qwen2.5-7b-instruct | answer | Chat → Vex | 2547 | 39.73% → 64.60% | -1.2782 / -0.3732 / 0.1939 | 29.48 / 52.13 / 12.41 | 0.371 | 482–554 |
| qwen2.5-7b-instruct | answer | Vex → Chat | 2547 | 39.73% → 49.48% | -0.5944 / 0.0389 / 0.1941 | 31.61 / 46.21 / 18.54 | 0.529 | 482–554 |
| qwen2.5-7b-instruct | context | Plain → HELIOS | 3047 | 59.68% → 76.72% | -3.1221 / -0.6613 / 0.0409 | 8.09 / 24.87 / 1.07 | 0.198 | 573–648 |
| qwen2.5-7b-instruct | context | HELIOS → Plain | 3047 | 59.68% → 74.62% | -2.7804 / -0.5235 / 0.0408 | 11.25 / 17.49 / 0.60 | 0.216 | 573–648 |
| qwen2.5-7b-instruct | answer | Plain → HELIOS | 3047 | 47.41% → 50.45% | -0.5438 / 0.1884 / 0.2353 | 14.81 / 39.26 / 23.59 | 0.692 | 573–648 |
| qwen2.5-7b-instruct | answer | HELIOS → Plain | 3047 | 47.41% → 75.47% | -2.1165 / -0.6384 / 0.2358 | 37.95 / 57.58 / 18.62 | 0.343 | 573–648 |
| qwen2.5-7b-instruct | context | Plain → Wren | 2765 | 58.00% → 76.22% | -3.0791 / -0.7130 / 0.0300 | 7.89 / 22.81 / 0.95 | 0.173 | 512–566 |
| qwen2.5-7b-instruct | context | Wren → Plain | 2765 | 58.00% → 74.21% | -2.7623 / -0.5799 / 0.0302 | 7.19 / 13.64 / 0.43 | 0.188 | 512–566 |
| qwen2.5-7b-instruct | answer | Plain → Wren | 2765 | 46.72% → 52.05% | -0.6667 / 0.1124 / 0.2014 | 12.72 / 37.66 / 17.85 | 0.602 | 512–566 |
| qwen2.5-7b-instruct | answer | Wren → Plain | 2765 | 46.72% → 73.11% | -1.9698 / -0.5819 / 0.2017 | 30.78 / 51.05 / 13.18 | 0.338 | 512–566 |
| qwen2.5-7b-instruct | context | Plain → Dana | 2641 | 61.02% → 77.00% | -3.1936 / -0.6348 / 0.0357 | 8.76 / 23.75 / 0.72 | 0.193 | 512–542 |
| qwen2.5-7b-instruct | context | Dana → Plain | 2641 | 61.02% → 76.39% | -3.0854 / -0.5921 / 0.0358 | 10.61 / 17.28 / 0.46 | 0.198 | 512–542 |
| qwen2.5-7b-instruct | answer | Plain → Dana | 2641 | 46.84% → 51.77% | -0.7148 / 0.0883 / 0.1730 | 10.12 / 32.02 / 11.69 | 0.590 | 512–542 |
| qwen2.5-7b-instruct | answer | Dana → Plain | 2641 | 46.84% → 75.71% | -2.4056 / -0.8103 / 0.1732 | 27.89 / 48.54 / 10.24 | 0.297 | 512–542 |
| qwen2.5-7b-instruct | context | Plain → Vex | 2547 | 60.14% → 77.62% | -3.3745 / -0.7428 / 0.0213 | 5.62 / 20.03 / 0.39 | 0.153 | 482–554 |
| qwen2.5-7b-instruct | context | Vex → Plain | 2547 | 60.14% → 76.26% | -3.1206 / -0.6416 / 0.0222 | 6.41 / 11.18 / 0.24 | 0.162 | 482–554 |
| qwen2.5-7b-instruct | answer | Plain → Vex | 2547 | 52.31% → 61.03% | -1.2198 / -0.0587 / 0.1352 | 8.97 / 29.96 / 6.97 | 0.458 | 482–554 |
| qwen2.5-7b-instruct | answer | Vex → Plain | 2547 | 52.31% → 74.40% | -2.3757 / -0.6103 / 0.1357 | 20.46 / 40.25 / 4.80 | 0.301 | 482–554 |
| qwen2.5-7b-instruct | context | HELIOS → Wren | 1432 | 23.11% → 46.03% | -0.5670 / -0.2050 / 0.1540 | 46.78 / 54.64 / 20.87 | 0.399 | 271–312 |
| qwen2.5-7b-instruct | context | Wren → HELIOS | 1432 | 23.11% → 46.50% | -0.5807 / -0.2155 / 0.1539 | 34.74 / 53.05 / 17.92 | 0.396 | 271–312 |
| qwen2.5-7b-instruct | answer | HELIOS → Wren | 1432 | 10.13% → 31.37% | 0.1301 / 0.2183 / 0.4028 | 71.58 / 75.24 / 63.87 | 0.597 | 271–312 |
| qwen2.5-7b-instruct | answer | Wren → HELIOS | 1432 | 10.13% → 21.99% | 0.2349 / 0.3124 / 0.4030 | 68.21 / 70.66 / 60.32 | 0.679 | 271–312 |
| qwen2.5-7b-instruct | context | HELIOS → Dana | 1391 | 28.92% → 50.08% | -0.6776 / -0.1917 / 0.1625 | 45.49 / 59.53 / 22.70 | 0.407 | 243–306 |
| qwen2.5-7b-instruct | context | Dana → HELIOS | 1391 | 28.92% → 49.78% | -0.6672 / -0.1842 / 0.1632 | 36.71 / 59.54 / 21.36 | 0.410 | 243–306 |
| qwen2.5-7b-instruct | answer | HELIOS → Dana | 1391 | 14.79% → 31.70% | 0.0824 / 0.2180 / 0.3729 | 61.40 / 69.16 / 57.28 | 0.609 | 243–306 |
| qwen2.5-7b-instruct | answer | Dana → HELIOS | 1391 | 14.79% → 30.80% | 0.0946 / 0.2283 / 0.3733 | 66.63 / 71.08 / 56.20 | 0.617 | 243–306 |
| qwen2.5-7b-instruct | context | HELIOS → Vex | 1315 | 22.39% → 47.45% | -0.6498 / -0.2799 / 0.1330 | 43.31 / 52.42 / 16.54 | 0.366 | 239–282 |
| qwen2.5-7b-instruct | context | Vex → HELIOS | 1315 | 22.39% → 46.09% | -0.6055 / -0.2456 / 0.1343 | 30.99 / 48.30 / 12.79 | 0.376 | 239–282 |
| qwen2.5-7b-instruct | answer | HELIOS → Vex | 1315 | 15.65% → 45.06% | -0.2380 / -0.0441 / 0.3198 | 58.63 / 66.99 / 45.29 | 0.485 | 239–282 |
| qwen2.5-7b-instruct | answer | Vex → HELIOS | 1315 | 15.65% → 24.71% | 0.0972 / 0.2386 / 0.3199 | 49.86 / 59.94 / 43.77 | 0.665 | 239–282 |
| qwen2.5-7b-instruct | context | Wren → Dana | 1338 | 8.44% → 34.48% | -0.2389 / -0.1345 / 0.1882 | 50.45 / 56.61 / 31.17 | 0.436 | 259–279 |
| qwen2.5-7b-instruct | context | Dana → Wren | 1338 | 8.44% → 33.83% | -0.2261 / -0.1228 / 0.1885 | 53.69 / 56.48 / 34.09 | 0.440 | 259–279 |
| qwen2.5-7b-instruct | answer | Wren → Dana | 1338 | 4.85% → 18.51% | 0.2662 / 0.3017 / 0.4019 | 68.04 / 70.13 / 59.38 | 0.668 | 259–279 |
| qwen2.5-7b-instruct | answer | Dana → Wren | 1338 | 4.85% → 26.04% | 0.1924 / 0.2315 / 0.4026 | 71.74 / 72.57 / 60.62 | 0.607 | 259–279 |
| qwen2.5-7b-instruct | context | Wren → Vex | 1282 | 12.14% → 40.27% | -0.4565 / -0.2796 / 0.1302 | 37.97 / 44.26 / 15.44 | 0.365 | 228–285 |
| qwen2.5-7b-instruct | context | Vex → Wren | 1282 | 12.14% → 39.62% | -0.4413 / -0.2662 / 0.1296 | 39.38 / 44.78 / 14.90 | 0.369 | 228–285 |
| qwen2.5-7b-instruct | answer | Wren → Vex | 1282 | 10.74% → 37.68% | -0.1140 / 0.0056 / 0.3056 | 60.45 / 64.45 / 40.89 | 0.504 | 228–285 |
| qwen2.5-7b-instruct | answer | Vex → Wren | 1282 | 10.74% → 24.28% | 0.0831 / 0.1815 / 0.3055 | 56.19 / 60.33 / 42.72 | 0.612 | 228–285 |
| qwen2.5-7b-instruct | context | Dana → Vex | 1306 | 16.71% → 42.32% | -0.4848 / -0.2366 / 0.1438 | 40.82 / 52.05 / 22.88 | 0.387 | 239–287 |
| qwen2.5-7b-instruct | context | Vex → Dana | 1306 | 16.71% → 42.08% | -0.4739 / -0.2276 / 0.1461 | 42.85 / 51.43 / 21.01 | 0.389 | 239–287 |
| qwen2.5-7b-instruct | answer | Dana → Vex | 1306 | 12.61% → 42.43% | -0.2051 / -0.0529 / 0.3071 | 59.80 / 66.20 / 42.44 | 0.482 | 239–287 |
| qwen2.5-7b-instruct | answer | Vex → Dana | 1306 | 12.61% → 23.14% | 0.0971 / 0.2111 / 0.3057 | 50.11 / 59.55 / 40.73 | 0.643 | 239–287 |

## Limits and provenance

- Training requires paired target representations.
- Story scaffolds differ with speaker and are not persona system prompts.
- Answer residuals include finite-five-rollout sampling variation; this test is not noise corrected.
- Scaling fits use target-training pairs, so these are calibrated relationships within each pair, not zero-shot transfer.
- Pair cohorts differ; answer residuals include finite-five-rollout noise.

The same full query appears in both prefixes, but surrounding narratives can add information. These remain story/framing comparisons, not isolated persona system-prompt interventions. Finite-five-rollout noise is not corrected, and this extension does not add behavioral or refusal measurements.

Strict parent: [published matched-query analysis](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/755cec2d8a0ecc9e5028aa6a4e8875d0a76527db/issue2054_section44_k5_gcp/transfer_calibration_v1/matched_queries/results/README.md). Parent result SHA-256: 3e4d958221bee06f23fa6f77a79ce8e7101671d662a16d7c6bf34baf574437a7.

Reproduce with `issue2054_k5_matched_run.py --affine --out eval_results/issue_2054/k5_matched_affine --inputs eval_results/issue_2054/k5_matched_offsets_strict/inputs.json`, then `issue2054_k5_matched_affine_plot.py`. Use a fresh output path, with the strict parent available in the sibling directory. Per-query errors/ranks, bias vectors, scalar coefficients, input hashes, code, monitoring and completion records are preserved in this publication.

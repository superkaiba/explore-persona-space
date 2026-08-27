# Matched-n map comparison — Qwen3.5-9B (thinking off) @ layer 18 (L*) vs Qwen2.5-7B-Instruct @ layer 19

All fits at matched n (train≈25k / val 400 / test 1,000), fp64 primal ridge,
λ val-selected; retrieval = P(true target in the k nearest neighbours of the
prediction) over the held-out test pool.

| quantity | Qwen3.5-9B (thinking off) @ layer 18 (L*) | Qwen2.5-7B-Instruct @ layer 19 |
|---|---|---|
| train rows (n) | 25000 | 25000 |
| hidden dim (d) | 4096 | 3584 |
| validation R² (selected λ) | 0.7273 | 0.7308 |
| held-out test R² | 0.7092 | 0.7251 |
| held-out transfer R² (WildChat) | 0.5992 | 0.6230 |
| selected λ | 1000.0000 | 3162.2777 |
| retrieval acc@1 (euclidean) | 0.7240 | 0.7710 |
| retrieval acc@10 (euclidean) | 0.9100 | 0.9240 |
| retrieval acc@1 (cosine) | 0.7410 | 0.7720 |
| retrieval acc@10 (cosine) | 0.9280 | 0.9260 |
| retrieval chance @1 | 0.0010 | 0.0010 |
| two-draw reliability ceiling (r) | 0.9090 | 0.9244 |
| floor: identity + learned bias R² | -1.6942 | -0.8943 |
| floor: identity (copy input) R² | -4.6794 | -2.5318 |
| floor: scaled identity R² | -0.0499 | 0.0777 |
| floor: shuffled pairing R² | -0.0198 | -0.0227 |
| floor: train mean R² | -0.0249 | -0.0256 |

Anchor gate: realized R² 0.7251 vs expected 0.7251 (|Δ| = 0.00e+00, tol 0.01).

Paired shared-test-row comparison: R²(9B@L*) = 0.7092, R²(7B@L19) = 0.7251, Δ = -0.0158 (95% CI [-0.0275, -0.0046]; verdict: h1_consistent).

# Superseded — 2,000-row run-length capture

Kept for provenance. **Do not consume these**: the live artifact one directory
up is the FULL-CORPUS (120,000-row) capture, which is the entire `sae_fit`
pool rather than a sample of it.

| | 2,000-row (here) | full-corpus (parent dir) |
|---|---|---|
| rows | 2,000 (seeded draw over the pool) | 120,000 (the whole pool) |
| `mean_run_length` finite | 103,217 / 131,072 | see parent meta |
| `template_token_frac` finite | 109,211 / 131,072 | see parent meta |
| complete-case (all covariates) | 101,621 | see `../covariate_refresh.json` |

Why it was superseded: the consumer's read mask is the INTERSECTION of finite
values across every covariate, so the 27,855 features this capture never saw
were dropped from EVERY predictor's read — not just the two run-length slots.

Produced by `scripts/issue1482_run_length.py` at commit 8636046e83; gates
row-occupancy 2.1538 / span-length 2.4297 / token-null 0.7309 (all inside the
10% band). Provenance detail is in `run_length_perfeature.meta.json` here.

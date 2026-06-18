---
name: random-bucket-persona-alignment
description: Unbiased FineWeb/LMSYS random buckets yield ~5% positive-cos hits for OOD personas (villain) — k=50 from a 400-doc sample fails assert_pool_size_meets_k. Planner must size k_random_bucket separately or provision ~10x docs.
metadata:
  type: feedback
---

Random-bucket arms (unbiased FineWeb+LMSYS samples) yield very few persona-aligned docs for niche/OOD personas: #375 measured **21/400 (~5%)** positive-cos docs for the villain direction at L20, vs easy k=50 from the curated persona-style pool. Expect `k_max ≈ corpus × 0.05` for niche personas and `≈ corpus × 0.5` for mainstream ones.

**How to apply:** when `assert_pool_size_meets_k` fires on a random-bucket/OOD arm, check the corpus-vs-persona alignment rate before bouncing — a facts-of-nature ceiling means the PLANNER must revise the k spec (separate `k_per_persona_random_bucket`, or ~10x more docs), not the implementer. Do NOT recompute persona directions to inflate the hit rate (defeats the unbiased-arm purpose). `--degraded-pool-ok` is acceptable only if the analyzer will flag the unequal sample size in the write-up.

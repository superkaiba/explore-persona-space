---
name: random-bucket-persona-alignment
description: Random-bucket sample (unbiased FineWeb/LMSYS docs) yields ~5% positive-cos hits for OOD personas like villain; plans specifying k=50 from a 400-doc random sample will fail assert_pool_size_meets_k.
metadata:
  type: feedback
---

When constructing example pools for in-context drift / persona-direction experiments, the "random-bucket" arm (an unbiased sample of FineWeb + LMSYS docs, by construction NOT pre-filtered by persona alignment) yields very few persona-aligned docs for OOD personas.

Concrete numbers from task #375 (Qwen-2.5-7B-Instruct, villain persona direction at L20, 400-doc random sample of 200 FineWeb + 200 LMSYS):

- **21 / 400 docs** (~5%) had positive cos against the villain direction.
- For comparison, the persona-style pool (1200 docs, top200 bucket) easily gave k=50 villain hits.

**Why:** The random-bucket sample is unbiased — by design ~50% of docs would have positive cos against ANY direction at chance. But for niche personas (villain, criminal, sycophant), aligned content is rare in mainstream web crawls. Most positive-cos docs in the random bucket are weakly aligned (cos near 0).

**Implications for planners:**

- For random-bucket arms in persona experiments, expect **k_max ≈ corpus_size × 0.05** for niche/OOD personas like villain, and **k_max ≈ corpus_size × 0.5** for mainstream personas like helpful-assistant.
- Specify `k_per_persona_random_bucket` separately from `k_per_persona` (the main persona-style pool). Don't reuse the same k.
- Or: provision a much larger random-bucket sample (e.g., 4000 docs to get k=200 villain hits at 5% rate).
- The unbiased nature of the random bucket is what makes it valuable as a P1 sensitivity arm — DON'T recompute persona directions to inflate hit rate (that defeats the purpose).

**Implications for experimenters:**

- When `assert_pool_size_meets_k` fires for a random-bucket / OOD-persona arm, check the corpus-vs-persona alignment rate before bouncing. If it's a "facts of nature" ceiling (e.g., 21/400 villain hits), the planner needs to revise the k spec, not the implementer.
- The `--degraded-pool-ok` flag is the right escape ONLY if the analyzer is prepared to flag the unequal sample size in the write-up.

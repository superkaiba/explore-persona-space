---
name: Check genealogy for evolutionary / search experiments
description: For any genetic / evolutionary / beam-search clean-result, trace the top-K candidates back to gen-0 ancestors BEFORE assigning confidence; single-root genealogy means the "search" is really a neighborhood scan
type: feedback
---

Before assigning HIGH confidence to any "search-discovered" claim (evolutionary search, beam search, GA, iterative refinement with multiple seeds), trace the top-K final candidates back to their gen-0 ancestors and count distinct lineages. Search loops routinely converge on one productive seed's neighborhood — the plan's `diversity_min_lineages=N` constraint fails in practice because runner-up lineages get out-competed within a few rounds.

**Why:** issue #331 — all 10 top obscure-only candidates descended from one Phase-0-seeded phrase (`apis papyrus est`); the mutation operator just explored first-word substitutions. Confidence dropped HIGH → MODERATE.

**How to apply:**
1. Load the genealogy/ancestry JSON (or reconstruct from per-round outputs); for each top-K candidate, follow `parent_phrase` back to round 0, collecting ALL ancestors for multi-parent operators like `llm_crossover` (parse parents out of `mutation_detail`).
2. Count distinct gen-0 roots feeding the top-K.
3. `n_roots == 1` → drop confidence one level, add an explicit caveat, reframe "search discovered X" → "search characterized the neighborhood of one productive seed", and call out the failed diversity constraint.
4. `n_roots ≥ 3` → diversity worked; confidence stays at whatever the data supports.

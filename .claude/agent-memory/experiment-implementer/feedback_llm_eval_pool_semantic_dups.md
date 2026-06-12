---
name: LLM-generated eval pools need semantic dup gates + frozen-once guards
description: Char-Levenshtein near-dup gates miss cross-family paraphrases in LLM-generated claim/eval pools; sampled generation phases are not idempotent and need a manifest-keyed rebuild guard.
type: feedback
---

Two traps when a P0-style phase assembles an eval pool with LLM generation (#612, 2026-06-12):

1. **Char-Levenshtein (<0.85) near-dup gates are NOT sufficient for LLM-generated pools.** Generation across phrasing families ("X, right?" vs "I'm pretty sure that X. Am I right?") produces the SAME misconception at char-sim < 0.85, and re-introduces excluded train-set claims in different words. Sonnet also keeps regenerating the same famous items (lightning/10%-brain/goldfish) unless the avoid-hint is TOPIC-SCOPED and includes previously-rejected candidates.
   **Fix pattern:** normalized-core (strip family wrappers/tag tails) + stemmed-Jaccard (suffix-strip + 6-char truncate, ≥0.34) prefilter → temp-0 Sonnet "same underlying claim? YES/NO" confirm, memoized per pair; apply at acceptance AND as a final assembled-pool assert. Also: a unanimous 3-vote Sonnet falsity audit still passes technically-true claims (longitude technicalities, recently-changed facts) — the implementer manual spot-check is load-bearing, keep a MANUAL_REJECT list with rationale.

2. **Sampled generation phases are not idempotent — guard the rebuild.** If a phase samples new rows at temp 1.0, an accidental re-run silently produces a DIFFERENT pool after the experiment locked onto the committed one. Add a frozen-once guard at entry: if output + sha manifest exist and match → log + exit 0; mismatch → raise; rebuild only behind `--force`. (Per-API-call checkpointing alone does NOT make the phase idempotent.)

**How to apply:** any claim-pool / question-pool / persona-pool audit-and-generate phase; any disjointness assert between LLM-generated and frozen text sets.

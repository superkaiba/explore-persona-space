---
name: check-paraphrase-pool-matches-chosen-fact
description: After a Phase-0 fact/entity switch, downstream-derived cache fields (paraphrase pools, specialist/workup fields) may keep the abandoned fact's values; verify via raw completions
metadata:
  type: feedback
---

When Phase 0 abandons one candidate fact/entity and switches to another, cache-invalidation often only refreshes top-level predicate fields (`entity`, `canonical_predicate`, `counter_predicate`) and skips downstream-derived fields (`canonical_paraphrases[...]`, per-mechanism specialist/workup/drug/imaging fields) cached from a generator call made with the OLD fact.

**Why:** task #407 (2026-05-31) — the orchestrator posted the invalidation marker, but paraphrase + mechanism fields kept the abandoned Creutzfeldt-Jakob values, so the NAGS-deficiency training data taught the model to emit CJD canonical text. Aggregates sat at floor (~0% canonical AND ~0% counter) — itself the tell that neither side of the planned predicate contrast was being matched.

**How to apply:** for ANY analysis of an experiment with a mid-run fact / entity / topic / probe switch, BEFORE trusting aggregates:
1. Sample 3-5 raw completions from each affected condition.
2. Verify the output text mentions the CORRECT entity and the emitted canonical/counter tokens match the chosen fact's predicates.
3. Verbatim abandoned-fact text in trained outputs = poisoned training data; downstream aggregates cannot speak to the planned hypothesis.

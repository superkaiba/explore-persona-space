---
name: check-paraphrase-pool-matches-chosen-fact
description: After a Phase-0 fact/entity switch, the paraphrase pool may not have been regenerated even when the canonical/counter predicate fields were updated; verify by reading raw completions
metadata:
  type: feedback
---

When an experiment's Phase 0 abandons one candidate (e.g. K2 firing on
the first fact-pick) and switches to a second candidate, the cache /
metadata file holding the per-mechanism paraphrase pool, specialist
fields, workup fields, drug fields, imaging fields, etc. may NOT be
fully regenerated even when the orchestrator claims to have done a
cache-invalidation step.

**Why:** Cache-invalidation logic often only invalidates per-key
top-level fields (`entity`, `canonical_predicate`, `counter_predicate`)
but skips downstream-derived fields like `canonical_paraphrases[0..9]`
that get cached from a generator call. If the generator's input was
the OLD fact, the cached output stays OLD.

**How to apply:** For ANY clean-result analysis of an experiment that
involved a mid-run fact / entity / topic / probe switch, do BEFORE
trusting the aggregate numbers:

1. Sample 3-5 raw completions from each affected condition (e.g. the
   trained models on the regime where the switch happened).
2. Verify the model's output text mentions the CORRECT entity, not the
   abandoned one.
3. Verify the canonical / counter answer tokens the model emits match
   the chosen fact's predicates, not the abandoned fact's.
4. If you see verbatim abandoned-fact text in the trained models'
   outputs, the training data was poisoned. The aggregate numbers
   downstream cannot be interpreted as evidence about the planned
   hypothesis.

Incident: task #407, 2026-05-31. The orchestrator posted
"invalidated regime_facts.obscure_real key" at 2026-05-28T20:52 and
proceeded with the new fact's fp-calibration to completion. But the
`canonical_paraphrases` / `counter_paraphrases` / per-mechanism
specialist/workup/drug/imaging fields kept their Creutzfeldt-Jakob
disease values from the abandoned fact-pick #1. The N-Acetylglutamate
synthase deficiency training data therefore taught the model to emit
CJD canonical text on NAGS probes. Caught at analyzer step by reading
raw completions; the aggregate numbers all sat at floor (~0% canonical
AND ~0% counter), which itself was the signal that nothing on the
trained side matched either side of the regime_facts predicate
contrast.

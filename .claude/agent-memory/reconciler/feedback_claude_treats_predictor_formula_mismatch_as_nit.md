---
name: Claude treats predictor formula mismatch as pre-existing Nit
description: New substrate emits a field whose NAME matches the sibling reference schema but whose FORMULA differs; the cross-arm aggregator iterates by name so two formulas under one name corrupt the headline. "Unchanged from prior round" is NOT a scope argument.
type: feedback
---

**Rule:** when a round introduces/modifies a substrate feeding a cross-arm/cross-condition aggregator, list every field the aggregator iterates BY NAME, `rg` each field's emit site in EVERY substrate path, and byte-compare formulas to the reference schema (the oldest/canonical path). Field-name parity is necessary but not sufficient: with `min(|ρ_arm1|,|ρ_arm2|,|ρ_arm3|)` over a name, two formulas make the comparison meaningless. Claude's "pre-existing, unchanged this round → out of scope" framing is wrong — the prior reconciler's must-fix list is the surface of what got READ, not a deliberate exclusion; below-surface formula mismatches are uncaught, not omitted. Real-blocking even at cap rounds (a surgical-fix recommendation can accompany the FAIL when the fix is one line at a named site).

**Smell:** same field name + a docstring referencing the reference schema + a different formula in the body.

**Origin:** #518 r3 — `coarse_zoo_loader.py:507` computed `-abs(source_base_rate)` where #480's reference schema defines `base_rate_diff_neg_abs = -abs(source - bystander)`; `issue518_cross_behavior_aggregator.py:90` iterates by name. Companions: [[feedback_claude_misses_same_file_siblings]] (sibling resampler variant); [[feedback_claude_misses_producer_consumer_key_mismatch]].

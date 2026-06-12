---
name: Claude treats predictor formula mismatch as pre-existing Nit
description: When round-N adds a NEW substrate loader emitting a predictor field whose NAME matches a sibling-substrate reference schema (#480) but whose FORMULA differs, Claude PASSes by walking the round-(N-1) must-fix table and tagging the formula mismatch as "pre-existing, unchanged this round → out of scope." Codex catches the cross-arm aggregator iterates the field by name, so two different formulas under one name corrupt the cross-arm headline. Defense: when the cross-arm aggregator's predictor list contains field X, open EVERY substrate's emit site for X and verify formula byte-equivalence to the reference schema. "Unchanged from prior round" is NOT a scope argument — the reconciler list was the surface of what got READ, not deliberately omitted. Origin: task #518 round-3 (`coarse_zoo_loader.py:507` `-abs(source_base_rate)` vs #480's `-abs(source - bystander)`; `issue518_cross_behavior_aggregator.py:90` iterates by field name).
type: feedback
---

When round-N integrates a NEW substrate-emit path (e.g. `coarse_zoo_loader.py` for the refusal/EM arms in #518) parallel to an existing sibling schema (#480's syco `_inputs/predictor_comparison.json`), Claude code-reviewer PASSes by:

1. Walking the round-(N-1) reconciler must-fix table and confirming each item lands.
2. Spot-checking that the new field NAMES match the schema.
3. Tagging any below-surface semantic mismatch as "pre-existing, unchanged this round → out of scope."

The trap is that field-name parity is necessary but not sufficient. When the cross-arm aggregator (e.g. `scripts/issue518_cross_behavior_aggregator.py:90`) iterates predictors BY NAME and computes `min(|ρ_arm1|, |ρ_arm2|, |ρ_arm3|)`, two different FORMULAS under one field name silently corrupt the cross-arm comparison: arm1's ρ measures one quantity, arm2/3's ρ measures another, the `min` is meaningless for that predictor cell.

**Codex catch pattern**: open the field's emit site in EACH substrate path and byte-compare the formula to the reference schema. In #518: #480 row 1 (`source=0.05`, `bystander=0.038`) carries `base_rate_diff_neg_abs = -0.012`, matching `-abs(source - bystander)`. The new refusal/em loader at `coarse_zoo_loader.py:507` computes `-abs(source_base_rate) = -0.05` — a different predictor (source-constant across bystanders).

**Claude's "out of scope" framing is wrong**: "unchanged from round-(N-1)" is NOT a scope argument. The round-(N-1) reconciler's must-fix list is the surface of what got READ during that reconciliation, not a deliberate exclusion. Below-surface formula mismatches are uncaught, not omitted.

**Why:** Cross-arm headline corruptions hide under field-name parity; the aggregator's predictor list is the actual contract surface for cross-arm comparability. Field-name parity gives the reviewer a false-PASS handhold ("the field is populated, the schema matches"). Memory companion: "Claude misses sibling resampler inconsistency" (#489 r-3) and "Claude misses producer/consumer JSON key mismatch (path vs inline)" (#514 r-2) — same shape, different surface (function-body math vs key-name vs formula).

**How to apply:** When a round-N introduces or modifies a substrate that contributes to a cross-arm/cross-condition aggregator, list every field the aggregator iterates by name, then `rg` each field's emit site in every substrate path, then byte-compare formulas. If any substrate's formula deviates from the reference-schema formula (the oldest / canonical / paper-replication path is usually the reference), that's Real-blocking even if "unchanged this round." Smell: the same field name + a docstring reference to a reference schema + a different formula in the body. The cap-3 question is separate — surgical-fix recommendation can accompany the FAIL when the strategy is converging (each round closed prior blockers + uncovered finer-grained ones, fix space is one-line, fix site is named).

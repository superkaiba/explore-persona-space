---
name: Fix-took gate must sit between the two signatures
description: A residual-bug detector threshold set above BOTH the bug signature and the clean signature is toothless; calibrate it between them (#585 float-identity gate)
type: feedback
---

A "fix actually took in THIS run" gate is only discriminating if its threshold sits BETWEEN the expected value under the bug and the expected value under the fix. #585 set its cross-fraction float-identity gate at "≥0.5 flags residual stale serving" — but #549's own evidence shows same-weights vLLM regeneration reproduces identical outputs only ~19–27% (the BUG signature), while distinct weights give ~0. Both hypotheses pass <0.5, so the gate can never fire, and a flat corrected curve becomes ambiguous between "parent mechanism wrong" and "bug recurred".

**Why:** The bug signature in stochastic pipelines is often a MODERATE identity/agreement rate (regeneration nondeterminism), not ~1.0. Thresholds borrowed from a "byte-identical" intuition miss it entirely.

**How to apply:** Whenever a plan registers a residual-bug / fix-took / stale-serve detector, look up the measured signature of the bug from the parent audit and the expected clean value; REVISE-or-concern depending on whether the raw diagnostic (e.g. per-pair identity heatmap) is persisted so the analyzer can re-threshold post-hoc. If persisted + benchmark committed elsewhere → Concern with the corrected decision rule spelled out; if the diagnostic is not persisted → Must-Fix.

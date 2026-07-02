---
name: Exact-value test constants — re-derive independently
description: Plan-specified hand-computed test constants (logP margins, LSE formulas) must be re-derived; a wrong constant in the SOLE numerics-pinning test is REVISE-grade. Uniform-logits fixtures pin nothing about span selection.
type: feedback
---

Re-derive every hand-computed exact-value assertion a plan specifies for its tests; do not trust the plan's formula. **Why:** #851 (infra promotion of the #722 tf-margin): the biased-logit test asserted `margin == (delta − LSE([delta,0×(V−1)])) − (−log V)`, but with the bias applied at every position the LSE term hits BOTH pools (pos = delta − LSE, neg = −LSE, margin = delta exactly). The wrong constant sat in the ONLY test pinning suffix-exclusion/span-boundary numerics — on failure an implementer may weaken to a sign check or fit the constant to output, letting a lift-introduced span bug through.

**How to apply:**
- Recompute pos/neg per-pool values AND the difference; prefer per-pool assertions (differences cancel shared normalization terms and pin less).
- **Uniform-logits fixtures pin NOTHING about span selection or suffix exclusion** — any subset of identical per-token values has the same mean. Only a token-position-asymmetric bias distinguishes wrong spans.
- LN (mean-vs-sum) is pinned only if ≥1 answer spans ≥2 tokens; single-token answers make mean == sum degenerate.

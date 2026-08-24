---
name: claude-interp-skips-derived-quantity-inversion
description: "#2479 r3: Claude interp verifies STATED caption numbers exactly but skips clauses that IMPLY a quantity (pool = 1/chance); invert/derive such clauses yourself; a point value at/below the committed distribution's min is a false claim, not rounding"
metadata:
  type: feedback
---

When a caption/prose clause states a quantity DERIVABLE from a committed
field (pool size from `acc1_chance = k/n_pool`, a rate from counts, an n
from a fraction), perform the inversion yourself against the primary
artifact — even when the Claude interpretation-critic's round was
otherwise exhaustive. #2479 r3: Claude verified the 15/16 retrieval
reversal, the chance ≈ 0.005 value, and the y-ordering on Figure 4 but
never inverted the 16 per-character chances; the caption's "about 205
candidates" was contradicted by the committed pools (206–249, mean
235.5) — Codex caught it, recount confirmed exactly, verdict REVISE.

**Why:** stated-value verification passes while the derived clause is
wrong; the writer's error came from inverting a 4dp-ROUNDED sibling
diagnostic (0.0049 → ≈204) instead of the verdict-exact field — so also
check WHICH artifact grain a derived number was computed from
(verdict-exact vs rounded diagnostics), cf.
[[codex-recount-with-silent-normalization]].

**How to apply:** (1) In interp/clean-result reconciles, list every
caption clause of the form "X per/of/across Y" where Y = f(committed
field); recompute f for ALL units, not the one nearest the claim.
(2) Rounding defense fails when the claimed point value sits at/below
the committed distribution's MINIMUM while most units are far above —
that is a false description of the population, not honest rounding
(here 15/16 pools were 227–249). (3) Severity: a factually wrong
derived number in a durable body is round-forcing at the interp bar
even when non-headline and direction-conservative (it made the result
look weaker); persist at CONCERN (blocks advance) — BLOCKER stays
reserved for production-crash class per workflow.yaml
reconciler_special_case.

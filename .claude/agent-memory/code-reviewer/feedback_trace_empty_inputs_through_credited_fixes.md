---
name: trace-empty-inputs-through-credited-fixes
description: Before crediting a fail-open fix or presence gate, feed it the empty/None/degenerate input and verify VALUES, not key existence — two verified misses in one round (#1739 cms r2)
metadata:
  type: feedback
---

When crediting a fail-open FIX or a presence/coverage GATE, run the degenerate
input through it symbolically before PASSing: the empty dict, the absent file,
the None value, the key-present-but-value-None row.

**Why:** #1739 cms round 2 (2026-08-22) — I PASSed a hardening commit with two
real defects the Codex twin caught, both the same class:
1. I verified `load_groups_map` was gated on file existence but never traced
   `groups_by_ctx={}` through the consumer: `groups_by_ctx.get(cid) not in
   overlap` retains EVERY context when the map is empty (`None not in set`),
   so the "overlap-excluded" recompute aliased to the full read and reported
   `status="ok"` — a vacuous verdict gate (Critical).
2. I verified a "substantive presence" predicate EXISTED but not what it
   validated: `"r2_map" in pl` passes `r2_map: None`, and the plan-required
   companion set (identity+bias, cosine+euclidean kNN) was never checked; the
   test fixture itself omitted kNN and I read it as adequate (Major).

**How to apply:** For every fix/gate I credit in a review:
- Enumerate its inputs and ask "what does this do on {} / [] / None / missing
  file / key-present-value-None?" — trace the CONSUMER, not just the guard.
- A `x.get(k) not in S` / `k in d` pattern is a red flag: `.get` returns None
  for unknowns and `in` checks keys, not values — both silently pass
  degenerate rows.
- A test fixture is part of the evidence surface: if the fixture omits a
  plan-required field and the gate passes it, the gate does not check the
  field — read fixtures as adversarially as code.
- Cross-check presence predicates against the PLAN's required field set, not
  just against "some check now exists" (the r1→r2→r3 ratchet here: key-only →
  non-empty-with-key → finite-full-companion-set).

Related: [[plan-first]] (the plan names the required companion set — the r2
predicate was credited against the CONCERN's wording, not the plan's).

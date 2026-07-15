---
name: resume/seed skip mechanisms starve downstream min-N gates
description: Any resume-skip or seed-consume mechanism changes the FRESH-row denominator — audit every downstream min-N assert/gate (wiring checks, equivalence gates) for the seeded case (#1335 r6)
type: feedback
---

When adding a resume-skip / seed-consume mechanism, sweep the pipeline for downstream gates that assert a MINIMUM count over freshly-produced units (wiring checks, equivalence gates, "needs >= 2 rows" asserts) and define the seeded-case semantics explicitly: skip-loud with a sidecar record when the shortfall is seed-attributable (the consumed units carry their original validation), assert as before otherwise — and make each gate's CONSUMER tolerate the skip record.

**Why:** #1335 att-20260715-180412 — the round-5 HF resume seed worked as designed, consuming a cell's prior captures; the fresh-row wiring check then asserted `>= 2 rows` and killed the run. Full-lane-only interaction, structurally unreachable in local smoke (seed was smoke-hermetic by design).

**How to apply:** grep the touched pipeline for `assert.*>=|min_|needs.*rows` after adding any skip/seed path; add a pinned fixture test for the seeded-shortfall branch AND the not-seeded branch; check the gate JSON's consumers for KeyErrors on the skip record.

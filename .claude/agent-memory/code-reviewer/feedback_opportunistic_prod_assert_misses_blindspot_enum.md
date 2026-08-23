---
name: opportunistic-prod-assert-misses-blindspot-enum
description: Revision rounds that add production-only asserts as opportunistic fixes routinely forget to add them to the smoke blind-spot enumeration — grep the round diff for new `if production:`-gated asserts and diff against the enumeration block.
metadata:
  type: feedback
---

Revision-round diffs that add `if production: assert ...` (or `if not
production: return` before a raise) as OPPORTUNISTIC fixes routinely update
every other surface (tests, blocker disposition, per-phase smoke sections)
but miss the ONE-LINE addition to the `## Smoke run` blind-spot enumeration
— the Step 0.71 trigger fires on exactly this shape ("assert only on the
production branch").

**Why:** #2476 r2 added a D2c full-overlap assert (g1 Minor-3 fix,
production-gated because the smoke's 24-row slice can only ever partially
overlap) and updated the round-2 enumeration for every PLANNED branch
(tiny-model, sae-dict, Hub legs, n_perm caps, stream cursor, G1/G4
demotion) — but not for the fix-born assert. The enumeration block tracks
the plan's anticipated branches; fix-born branches have no plan row to
mirror, so nothing prompts the update.

**How to apply:** on any revision-round diff, grep the ROUND diff for
`if production` / `if not production` / `_production(args)` gating a new
assert/raise/return, then grep the marker's enumeration block for each
hit. An unenumerated hit is `smoke-blind-spot-unenumerated` (FAIL per Step
0.71) even when everything else in the round is clean — the fix is 1-2
enumeration lines in a marker re-post, so name every sibling in one round.

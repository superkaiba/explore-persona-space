---
name: verify check 11b collides distinct "N of 23 cells" rate phrases
description: planned-vs-actual denominator check FAILs when two different rate-style "X of 23 cells" claims coexist; phrase rate counts as "X of its 23 bystander cells" to dodge the regex
type: feedback
---

`verify_task_body.py` check 11b (planned-vs-actual denominator consistency) pairs ANY two `X of N <noun>` claims with the same N + noun stem (nouns: cells/conditions/seeds/sources/domains/...) and FAILs if the numerators differ — even when both are rate-style reports of different quantities, not scope claims.

**Why:** Incident #480 round-2 re-fold (2026-06-11): new prose "21 of 23 cells beyond 1 nat" (logP↔logit divergence count) collided with pre-existing "14 of 23 cells pinned at zero" (saturation pathology) — FAIL, though neither was a planned-vs-actual claim.

**How to apply:** When writing a rate-style count over a panel, break the regex by inserting a possessive or non-tracked adjective: "21 of its 23 bystander cells" / "14 of the source's 23 cells". The regex only matches `\d+ of \d+ (swept|planned|matched|testable|tested)? <noun>` with the noun immediately following. Reserve the bare "X of N cells" form for genuine coverage/scope claims.

---
name: revision-stale-gate-declaration-figures
description: When a revision re-derives an API/wave estimate in §9, grep §7 gate declarations + §11 rationale for the OLD figure — pilot-gate wave_n_calls declarations are copied verbatim across versions and go stale
metadata:
  type: feedback
---

When a plan REVISION re-derives a call-count / wave-size estimate in §9
(e.g. an S2-style scope cut shrinking a judge wave), grep the §7 gate
declarations AND §11 Decision Rationale for the PRIOR version's figure
before passing the round: `judge_pilot_gate` wave declarations
(`wave_n_calls≈...`) and rationale mentions ("the one ≈Nk wave") are
copied verbatim from the prior version and silently keep the stale count.

**Why:** found live in #2329 `q35_ladder_decay` v5→v7 (2026-08-19): §9
re-derived Leg B production 7.5k→4.7k, but G4b still declared
`wave_n_calls≈7.5k` and §11 still said "the one ≈7.5k wave". Inert there
(threshold_base=0 pinned forces batch regardless, and the rule-26
satisfiability guard keys on arms/draws, not wave_n_calls) — but rule 26
requires the declaration to mirror the wave's dispatch kwargs 1:1, and an
UNPINNED declaration inside the OTPM-probe region would make the stale
count routing-relevant.

**How to apply:** in round-2+ PLAN MODE, after verifying a re-derived §9
estimate reconciles, `grep -n '<old figure>' vN.md` on the CURRENT version
— any hit in §7/§11 is a mechanical nit (blocking only if the declaration
is unpinned and the stale count changes the computed route). Related:
[[revision-row-redistribution-check]], [[compound-wall-cell-parse-check]].

**Closure-verification nuance (v8 round 3, verified CLOSED):** when
checking the fix landed, two figures legitimately DIFFER — the declared
`wave_n_calls` is the PRODUCTION item count (pilot excluded; it mirrors
the production dispatch kwargs), while the §9 Leg/phase CEILING includes
the pilot (#2329 v8: G4b declares ≤≈6.7k full-survival production while
§9 books ≈7.1k incl. the 416-call pilot — both correct, 7,096 − 416 =
6,680). Don't flag that delta as a fresh mismatch. Surviving hits of the
old figure in a changelog / self-check line describing the fix are
meta-references, not stale declarations.

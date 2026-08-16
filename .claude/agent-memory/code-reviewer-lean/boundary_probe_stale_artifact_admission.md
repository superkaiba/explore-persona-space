---
name: boundary-probe-stale-artifact-admission
description: A threshold/at-cap probe whose admission is pure count arithmetic is vacuously satisfiable when its own probe artifact PRE-EXISTS from a crashed round — require the probe commit state to be committed-this-round, not landed (#2321 R2 g2)
metadata:
  type: feedback
---

An evidentiary boundary probe (e.g. #2321 §3.6: bring the repo to EXACTLY
the 1,000,000-file cap with `probe_a`, then test whether a net-zero commit
is accepted AT the cap) gated only by count arithmetic
(`fresh + 1 == CAP` plus `fresh == expected`) can mint a false "confirmed":
if `probe_a` survives a crashed prior round, today's `fresh` COUNT INCLUDES
it, both admission checks pass, the idempotent probe-first commit returns
state `"landed"` without adding a file, and the whole A→B sequence runs one
BELOW the boundary — B is then trivially accepted under either semantics
and certifies nothing.

**Why:** probe-first/idempotent commit loops deliberately return `landed`
for pre-existing identical content, so "commit A ran" is NOT "commit A
moved the count". The count arithmetic is self-consistent in the stale-
artifact world; only the artifact's FRESHNESS breaks the symmetry.

**How to apply:** whenever a review covers a threshold/cap/boundary probe,
ask: can the probe's own artifact pre-exist (crash residue, prior round)?
If the commit helper distinguishes `committed` vs `landed`, the admission
must require `committed` (created THIS round) — or explicitly delete stale
probe artifacts and recount before admission. Check the verdict/rc route
for the stale case too (should be the unsettled/recompute route, never
success). Blast-radius framing for severity: if the false GO's downstream
failure is fail-loud (a rejected real commit → global stop) and no deletion
path is touched, it is a Major/CONCERNS, not a FAIL.

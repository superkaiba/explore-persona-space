---
name: loopvar-tmp-interpolation-evades-2336-lint
description: "#2336 shared-tmp lint arms E/F presume trailing/post-.tmp_ interpolation is process-varying; a loop-variable interpolant (.tmp_top25_{fam}) evades the lint but is still fixed-per-unit — same race class on duplicate launches"
metadata:
  type: feedback
---

When reviewing #2336-class atomic-write migrations, do not treat a clean
`--check-shared-tmp-name` run as proof the file is race-free. The lint's
arms E/F exempt f-strings whose interpolation sits after `.tmp` / directly
after `.tmp_` (trailing interpolation presumed process-varying). A name like
`f".tmp_top25_{fam}.npz"` evades every arm, yet `{fam}` is a LOOP variable:
within one fam the temp name is fixed, so concurrent duplicate launches on
one out-root hit the same [[fixed-name-tmp-atomic-write-fanout-race]] class
(#2329/#2546).

**Why:** #2552 r1 g4 — the commit migrated exactly the 4 lint-hit sites and
left 2 loop-variable-named writers that the lint structurally cannot see;
only a manual `grep '\.tmp_'` on the file surfaced them.

**How to apply:** on any savez/tmp-migration commit, grep the touched file
for ALL `.tmp` writer shapes (not just lint hits), then classify each
interpolant: process-varying (pid/uuid/rank) = safe; loop/config variable =
lint-invisible residual — note it with cross-commit attribution and suggest
migration. Also verify waiver placement one-per-hit-line at i-1 (stacked
waivers go inert at i-2, #2330).

Third evasion channel (#2564 r1 g7): a round-authored driver DELEGATES its
JSON writes to a reused parent module's fixed-name writer
(`import issue2162_run as R; R._write_json_atomic(...)`) whose FILE sits on
`SHARED_TMP_LEGACY_ALLOWLIST` — the class-sweep fixed every round-authored
writer BODY and the lint passed clean, yet 7 new destinations write through
the grandfathered `.name + ".tmp"` shape. Sweep round files for `R.<writer>`
/ cross-module write-helper CALLS too, then check the callee's file against
the allowlist (workflow_lint.py `SHARED_TMP_LEGACY_ALLOWLIST`). Single-process
caller = Minor (duplicate-launch-only hazard) with the cheap fix "rebind to
atomic_io.write_json_atomic"; a fan-out caller = the
[[fixed-name-tmp-atomic-write-fanout-race]] FAIL class. Related:
[[reused-module-internal-consumer-sweep]].

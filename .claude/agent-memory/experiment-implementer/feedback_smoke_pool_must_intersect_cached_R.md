---
name: Smoke question pool must intersect the on-policy R cache (or it builds 0 rows)
description: A per-cell training-mix builder keyed by question text silently builds 0 rows when the smoke question pool's questions are absent from the (production-keyed) on-policy R cache; the skip-if-cache-exists guard on a shared cache path causes the collision.
type: feedback
---

When a behavior-implant builder constructs training rows by keying an on-policy
response cache by QUESTION TEXT (`pos_r = read_cache(...); for q in questions: if
q not in pos_r: continue; rows.append(...pos_r[q]...)`), a smoke run that uses a
tiny hardcoded smoke question pool (`SMOKE_QUESTIONS`) will silently build **0
rows** whenever the R cache on disk was generated for the PRODUCTION question set
and the smoke questions are not among its keys. The non-smoke path usually
asserts the row count; the smoke path silently writes an empty file.

Root mechanism (issue #664 r15, `issue664_dispatch.py` + `issue664_build_training_data.py`):
the p0 elicitation guards each R cache with `if (CACHE_ROOT/"marker_R"/f"{src}.json").exists(): continue`.
Smoke and production SHARE the same `marker_R/<src>.json` path, so on a pod that
already ran production elicitation (r1-r6), smoke p0 SKIPS regeneration and the
4 smoke questions are absent from the production-keyed cache → builder
`continue`s every row → `wrote ... (0 rows)`. Then p1 training on 0 rows is a
no-op / crash.

**Why:** the skip-if-exists guard is keyed on a path shared between smoke and
production, but the cache CONTENT (which questions are keyed) differs by mode.
A shared cache path across two question regimes is the collision.

**How to apply:**
- To run a clean end-to-end smoke on a pod carrying production R caches: back up
  the production `<ctx>.json` cache files, `rm` them so smoke p0 regenerates them
  keyed by the smoke questions, run the smoke, then RESTORE the production caches.
  (Verify restore by file size — production marker_R is ~1.1MB/ctx, smoke ~10KB.)
- The durable fix (out of a launcher-restore scope, file as infra): NAMESPACE the
  smoke cache (`marker_R_smoke/`) so smoke and production never collide on the
  same path, OR make the smoke builder assert `len(pos_rows) > 0` (fail loud on an
  empty smoke mix instead of silently writing 0 rows).
- General rule: any "skip if cache file exists" guard on a path SHARED between
  smoke and production is a latent 0-row trap whenever cache CONTENT is
  mode-dependent. Either namespace the path by mode or assert non-empty output.

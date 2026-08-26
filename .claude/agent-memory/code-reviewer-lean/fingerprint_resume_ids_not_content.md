---
name: fingerprint-resume-ids-not-content
description: a fingerprinted resume predicate hashing ID SETS + params (fam/tag/draws/seed/panel-id sha/names) still resumes stale null/permutation caches when upstream DATA VALUES change under identical ids; certify resume fixes with a live two-run phase probe
metadata:
  type: feedback
---

A resume-predicate fix that fingerprints the generating REGIME (params, seed
key, step, candidate names, sha over panel-ID bytes) is a real fix for the
"no resume predicate" / presence-skip classes — but it is still an ID-level
key, not a content key: null-draw matrices derive from the DV residuals, so
an upstream recompute that changes r2/y VALUES while leaving panel ids, seed,
and names identical resumes stale null bands against fresh observed stats.
Grade the fix PASS + Minor (ask for one content hash of the derived-from
array, e.g. y_rank bytes), not FAIL — reachability is narrow because regime
wipes usually shift counts → panel ids.

**Why:** #2552 r2 g2 — `_draw_store` fingerprint closed the codex
ladder-restartability blocker; the residual content gap is the same family
as [[size-match-resume-skip-npz]] (size ≈ presence) one level up
(ids ≈ content).

**How to apply:** (1) whenever a reviewed diff adds a fingerprint/manifest
resume key, enumerate what the persisted artifact DERIVES FROM and check
each input appears in the key as content, not just identity; (2) certify
resume fixes LIVE, not only by unit test: run the phase twice in one scratch
out-root, grep the resume log line, and read the per-step `resumed_*` flags
in the output JSON (here: 108/108 resumed, byte-identical null stats);
(3) same round confirmed [[../code-reviewer/feedback_opportunistic_prod_assert_misses_blindspot_enum]]
again — fix-born `args.smoke or ...` asserts vs the plan's "No other
production gate is downgraded" closing literal; grep the round diff for new
smoke-bypassed asserts every fix round.

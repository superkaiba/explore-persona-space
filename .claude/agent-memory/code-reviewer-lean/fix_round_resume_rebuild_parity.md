---
name: fix-round-resume-rebuild-parity
description: Recipe for regression-directed fix rounds claiming resume-time aggregate rebuild + a new HALT gate (#2552 r2 g1) — both-shapes kill, merge-condition parity, unconverted-site sweep, stale fail-report
metadata:
  type: feedback
---

When a fix round claims "aggregate rebuild from persisted raw" plus a new hard HALT gate, verify five things (all bit in #2552 r2 g1):

1. **Both r1-named failure shapes killed separately** — an empty-aggregate WRITE shape (guard falls through and writes empties) and a never-written shape (early return) need different fixes; check each phase's realized branch, not the helper alone.
2. **Merge-condition parity, token-by-token** — diff the resume helper's overlay merge condition against the in-band merge in the dispatch path (`rec valid OR existing not valid`). Any asymmetry silently diverges resumed aggregates from fresh ones.
3. **Sweep for unconverted sites** — grep the raw-reduce idiom (`reduce(load(raw))`) after the fix; every remaining hit outside the new helper is a missed sibling (also catches smoke/calibration paths).
4. **Stale-overlay edge** — on a post-HALT re-run, confirm cache-served valid results land in the rewritten BASE raw and the overlay condition never demotes a valid base entry under a stale non-valid reissue record; then a stale fixed-name reissue file is harmless.
5. **HALT-gate residue** — a gate that writes a `<x>_fail_<unit>.json` report then raises must unlink it when the unit later PASSes; otherwise a fail report sits beside a passing done-marker (flag as Minor). Also verify the raise fires BEFORE the done-marker write (resume key) AND before any consumer aggregate write, and that expected-but-missing items enter the denominators (`n_valid/n_items` with missing counted) and the reissue set (they must still be present in the dispatch `items` list for the reissue filter to find them).

**Why:** the r1 Major (g5 M1) named two shapes; the fix was correct only because the helper mirrored the in-band merge exactly and all seven sites converted — each check above is a place a plausible fix would have silently regressed.

**How to apply:** any fix-round brief whose claims include resume/rebuild, completeness/floor gates, or reissue sets. Related: [[size-match-resume-skip-npz]], [[new-dial-missing-from-resume-regime]] (a halt-only flag correctly stays OUT of the regime hash when it changes no data — check it's recorded in meta instead).

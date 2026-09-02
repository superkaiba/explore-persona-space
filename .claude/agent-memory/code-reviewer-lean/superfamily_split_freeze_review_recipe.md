---
name: superfamily-split-freeze-review-recipe
description: Reviewing a frozen dev/test split built on a dedup/superfamily graph — edge-type x node-type reachability matrix, tautological partition guards, blocking-flag probe, unresolved-corpus disposition traced to the downstream gate
metadata:
  type: feedback
---

Recipe for reviewing a frozen split-manifest builder over a union-find
superfamily graph (#2658 group A, 2026-09-02):

1. **Edge-type × node-type reachability matrix.** List every edge criterion
   (problem-key identity, exact-text, near-dup, rephrase) and every node kind
   (free-text bank, keyed/composed, id-only benchmark, extraction). Check
   which pairs each edge can actually connect: in #2658, exact/near-dup/
   rephrase edges filter on `problem_id is None`, so a FREE-TEXT extraction
   corpus can NEVER link to a KEYED frame item — the extraction-overlap
   "measurement" for such a row is structurally inert (always 0), not
   measured. The tell: `n_barred_superfamilies == 0` on a row whose frame
   kind and extraction kind sit in disconnected node classes.
2. **Partition guards are tautologies.** `assert_disjoint(dev, test)` where
   both sets partition ONE dict by value can never fire; the real leakage
   risk is MISSED EDGES (near-dups in different superfamilies straddling the
   split), which no post-hoc set check sees. Don't credit the guard as
   coverage; check edge coverage instead.
3. **Blocking/approximation flags: read the committed artifact.** A disclosed
   cost cap (length-band blocking above N items) narrows the criteria when it
   fires — verify per-row `used_*` flags in the frozen manifest to decide
   latent vs realized (in #2658 all false: keyed/id items bypass the lexical
   pass entirely, keeping free-text pools under the cap).
4. **Unresolved-input fail-open: trace to the downstream gate before grading
   severity.** build_row swallowed ExtractionCorpusUnresolvedError into
   `extraction_resolved: false` + rc=0 freeze; severity dropped from Major to
   Minor because unit 8's registered launch gate (`clean_dependency_graph`,
   scripts/issue2658_power.py) FAILs on any eligible-but-unresolved row.
   Grep downstream consumers of the flag before writing the blocker.

**Why:** all four probes settled brief-mandated questions in one pass;
the inert-measurement mechanism (probe 1) is invisible to tests that only
assert guards raise.

**How to apply:** any diff freezing dev/test splits, dedup families, or
exclusion sets from a similarity graph; also [[fingerprint-resume-ids-not-content]]
for the sha-key stability half (key on generating parameters / file-loaded
bytes, never recomputed floats — #2658 passed this cleanly).

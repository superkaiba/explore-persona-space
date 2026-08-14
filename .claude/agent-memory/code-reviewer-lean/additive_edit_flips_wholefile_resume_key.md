---
name: additive-edit-flips-wholefile-resume-key
description: On "additive-only" edits to a fingerprint-keyed staged driver, check whether the edited FILE participates in upstream stages' resume keys — a whole-file code fingerprint silently invalidates every completed checkpoint under the default --stage all (#2222 g4 R1)
metadata:
  type: feedback
---

When a commit claims an "additive, no behavior change" edit to a multi-stage
driver whose checkpoints are fingerprint-keyed, grep the driver for its
code-fingerprint helper and check WHICH files it hashes. A whole-file
fingerprint (e.g. `files_fingerprint([reduce.py, analysis.py])` inside
`_percell_key`) means ANY edit to that file — even one scoped to a later
stage — falsifies every earlier stage's sidecar key, so the default
`--stage all` re-run recomputes all completed units ("stale checkpoint —
recomputing") and may re-stage their staged inputs.

**Why:** #2222 split-review g4 round 1 (2026-08-10): unit 3 added
`dataset_values` persistence to `stage_aggregate` only; the brief said
"fingerprint/resume contract untouched" — true of the MECHANISM, but
`reduce_code_fingerprint()` hashes the whole reduce.py, so all 24 percell
checkpoints' keys flipped. Correctness-safe (conservative recompute), but a
spurious 24-cell recompute + capture re-staging at relaunch unless the
operator runs `--stage aggregate` alone (`_load_percell` did no key check).

**How to apply:** on any review of an edit to a stage-checkpointed
`scripts/issue<N>_*.py`: (1) find the resume-key builder; (2) if it hashes
whole files that include the edited one, state in the verdict which
completed checkpoints the commit invalidates and which stage invocation
avoids the recompute; (3) distinguish "resume mechanism untouched" (a true
claim about code) from "resume keys unchanged" (usually false after any
edit). Related genre: [[gate-threshold-vs-shard-config]] (config change
silently disarming a keyed mechanism).

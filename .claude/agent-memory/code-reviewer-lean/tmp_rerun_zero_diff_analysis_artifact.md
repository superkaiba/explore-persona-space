---
name: tmp-rerun-zero-diff-analysis-artifact
description: Reviewing a 0-GPU analysis round that commits a results JSON — rerun the committed script to /tmp and diff against the committed artifact; a zero-diff is the single strongest check (#2225 fu1 residual-probe)
metadata:
  type: feedback
---

When a free-analysis round commits BOTH the analysis script and its output
JSON, and the staged inputs still exist on disk, rerun the committed script
end-to-end with `--out /tmp/... --skip-figure` (thread-caps prefix inline)
and recursively diff against the committed JSON, excluding only the
`reproducibility` block. A zero-diff certifies in ONE command what would
otherwise take four separate reads: (1) the in-code parity/assert gates pass
live, (2) the "mechanical verdicts" were computed by the code, not
hand-typed, (3) the committed artifact matches the committed code (no
post-run hand-edit), (4) the run is deterministic at the pinned seed. Cost
in the driving case: 43 s wall on a ~2k-row Gram-probe battery.

**Why:** #2225 fu1 residual-probe review (commit 5b42e44f): the rerun
produced 0 numeric diffs across ~700 JSON lines, retiring the banked-parity,
pre-registration-threshold, and NaN questions simultaneously; reading alone
could not have excluded a hand-edited verdict field.

**How to apply:** only when inputs are ALREADY staged (check paths first —
never trigger a download; pods.md ~10 GB download rule) and projected wall
is small (the committed JSON usually records `wall_s` — trust it as the
estimate). Write outputs to /tmp, never over the worktree's committed
artifacts. Exclude provenance/wall-clock blocks from the diff. Sibling:
[[stats-reuse-driver-live-probes]] (synthetic probes for imported helpers —
use those when inputs are NOT staged or the rerun is expensive).

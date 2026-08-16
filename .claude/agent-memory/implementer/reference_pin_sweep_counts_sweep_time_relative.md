---
name: pin-sweep-counts-sweep-time-relative
description: Recount a disputed pin-sweep fence against its OWN contemporaneous map TSV first — fence≠TSV means a stale/miscopied paste, not a second legitimate object; re-derive fresh and publish the internally consistent set
metadata:
  type: reference
---

When a reviewer disputes a prior round's Gate-scope pin-sweep count (#1288
block), there are TWO distinct quantities and both may be "right":

1. the disputed marker's OWN fenced block row count (recount it directly —
   `awk` the rows between its fences from the canonical events.jsonl note,
   `sort -u | wc -l`); and
2. a FRESH `select_step9c_tests.py --map-files` sweep at the current tip.

When the two derivations disagree, suspect the PASTED FENCE before the
sweep. #2321 r4 measured: the 41- and 43-path difflists (differing only
by 2 agent-memory `.md` files) produced BYTE-IDENTICAL 191-pair
`--map-files` TSVs — the same 157-test set — while the disputed v3 fence
held 158 rows with 26/25 asymmetric membership against its OWN
contemporaneous TSV. The divergent object was the hand-pasted fence
(stale/miscopied), not a second sweep result. Mechanism bound:
`--map-files` consumes the given path list and never runs git (no
fetch — the `select_step9c_tests.py` mapping-mode contract), so
`origin/main` moving between sweeps can change only the DIFFLIST INPUT,
never a mapping run's output for a fixed difflist; here the measured
input delta produced zero output change. Always recount a disputed fence
against `cut -f1 <its own map.tsv> | sort -u` before attributing drift.

**How to apply:** in the re-roll marker report BOTH derivations with
methods — the disputed block's recount (confirming/refuting the reviewer)
and the fresh set (which becomes the published block, internally
consistent: stated count == listed rows == `wc -l` of the col-1 dedup
union). Never copy either prior number; never present the fresh count as a
recount of the old block. The Step 9c gate re-runs the live selector at
merge anyway, so membership drift between rounds is context, not a defect.

Related: [[worktree-lint-grandfather-sync-lag]] (same root cause family:
worktree surfaces lag a moving main).

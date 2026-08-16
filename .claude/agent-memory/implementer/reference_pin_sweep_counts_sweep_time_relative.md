---
name: pin-sweep-counts-sweep-time-relative
description: Branch-scope pin-sweep hit sets/counts are sweep-time-relative — re-derive fresh at the current tip AND recount the disputed block; both prior numbers can be right for different objects
metadata:
  type: reference
---

When a reviewer disputes a prior round's Gate-scope pin-sweep count (#1288
block), there are TWO distinct quantities and both may be "right":

1. the disputed marker's OWN fenced block row count (recount it directly —
   `awk` the rows between its fences from the canonical events.jsonl note,
   `sort -u | wc -l`); and
2. a FRESH `select_step9c_tests.py --map-files` sweep at the current tip.

They can disagree in MEMBERSHIP even when the difflist is near-identical:
the diff base is FETCHED `origin/main`, so main moving between sweeps
shifts which regions of the branch's `.claude/**` prose surfaces differ
(whole prose-pin test families flip in/out), and the worktree's selector
copy can sit commits behind main's. #2321 r4 measured: 41-vs-43-path
difflists differing only by 2 agent-memory `.md` files produced 158- and
157-file hit sets with 26/25 asymmetric membership.

**How to apply:** in the re-roll marker report BOTH derivations with
methods — the disputed block's recount (confirming/refuting the reviewer)
and the fresh set (which becomes the published block, internally
consistent: stated count == listed rows == `wc -l` of the col-1 dedup
union). Never copy either prior number; never present the fresh count as a
recount of the old block. The Step 9c gate re-runs the live selector at
merge anyway, so membership drift between rounds is context, not a defect.

Related: [[worktree-lint-grandfather-sync-lag]] (same root cause family:
worktree surfaces lag a moving main).

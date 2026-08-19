# Dropped plan arm: check for a zero-GPU filter before documenting the drop; HF pre-overwrite revision bisect

Issue #2223, clean-result-critic round 1 (2026-08-14).

## Lesson 1 — a "silently dropped" plan condition may be a FILTER over committed artifacts

Lens 13 flagged the plan-committed published-topic subset (`A_pubtopic`) as silently dropped.
It was never a run cell: the driver realized the published topic as slot 0 of every persona's
topic list, so the subset is exactly the conversations whose id ends `__t0` in the ALREADY
COMMITTED trajectory JSONs — a zero-GPU filter, computed in minutes
(`scripts/issue2223_r5_pubtopic.py`). Before taking the After-Every-Experiment item-8
document-the-drop escape, grep the driver for how the plan condition was REALIZED (a slot
convention, an id suffix, a metadata flag) — subset arms are often derivable, not re-runnable.
Report tiny-n reads as directional isolation with n on every number; the rig's verdict rule
(MIN_SAMPLES-style floors) is stated inapplicable, never silently applied or skipped. A domain
with zero alive late cells is named (no zero bar), and the partial read still reported.

## Lesson 2 — pin a pre-overwrite HF revision by oid bisect, then verify by CONTENT

When an HF path was overwritten in place (two legs sharing one prefix), find the pre-overwrite
revision mechanically: `list_repo_commits` → restrict to the run window → bisect
`get_paths_info(..., revision=...)` on the file's blob oid vs the pinned (post-overwrite) oid;
the last commit with a different oid is the pre-overwrite revision. Then download THAT revision
and verify by content (the recovered #2223 copy's `meta.timestamp_utc` fell in the 7B leg's
window) — size alone does not prove which leg wrote it. Datetime gotcha: `str(commit.created_at)`
uses a SPACE separator, so string comparison against `"...T..."` bounds silently mis-filters.
Bonus: a recovered raw pool makes the previously-impossible CJK scan cheap — run it and replace
the "no scan possible" disclosure with the count (0/500 here; a nonzero count would need care
not to disturb settled numbers — report as an additional robustness read).

## Lesson 3 — re-render only the flagged figures; restore the rest

Re-running a multi-figure script to fix two legends rewrites every figure's bytes. Commit only
the flagged pair by explicit path and `git -C "$WT" checkout -- <other figure paths>` (worktree,
targeted paths) so the untouched figures keep their pinned SHAs.

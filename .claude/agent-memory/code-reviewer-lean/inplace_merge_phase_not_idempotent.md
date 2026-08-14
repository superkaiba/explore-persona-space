---
name: inplace-merge-phase-not-idempotent
description: A remediation phase that APPENDS recovered draws into per-unit partials in place, keyed on a never-reset censoring counter, double-appends on every re-run (cache serves identical scores) — check merge phases for resume idempotency (#2225 R1 g4)
metadata:
  type: feedback
---

A rule-28-style sync re-issue (or any recovery/merge phase) that mutates
per-unit partial JSONs IN PLACE is a resume trap when (a) its target selector
reads a counter the merge never resets (`rollout_n_api_refusal > 0` stays
true forever), and (b) its recovered values are appended to the kept-draw
list. A re-run — the natural move after a crash partway through a multi-hour
sync wave — re-selects the same targets, the rubric-keyed judge cache serves
byte-identical scores, and the SAME draws are appended a second time: means
shift toward the recovered draws with no error anywhere (rule 24(ii)'s
"duplicated draw masquerading as a recovery").

**Why:** #2225 R1 g4 (`issue2225_judge.py::run_sync_reissue`): targets from
`n_api_refusal > 0`, merge appends `sync_scores` to `rollout_draw_scores`,
block written in place per unit — every completed unit gets corrupted by the
rerun that recovers the crashed tail.

**How to apply:** for every phase that mutates existing partials in place,
ask "what happens on the second run?" — demand a done-marker the merge itself
writes (e.g. skip units whose `judge_meta.<reissue-key>` exists), or a
replace-not-append merge keyed on reissued item ids. The GOOD fix shape
(verified #2225 r2, `2c48d9e026`): done-marker written atomically IN THE SAME
`_atomic_write_json` as the merged draws (crash-mid-unit ⇒ unmarked+unmerged),
skip fires before target selection AND judge dispatch, and any spend-bearing
follow-on (parity re-judge) is gated on this run's recovered count so a full
rerun spends nothing; pin with a run-twice real-body test asserting stable
draw-list length + zero extra judge calls. Sweep siblings: append-
JSONL phases with last-wins dedup on read are fine; fingerprint-keyed
skip-if-done phases are fine; the in-place appender is the one shape that
corrupts. ALSO sweep sibling RESUME KEYS when one gets fixed: #2225 r2 pinned
the probe-application done-set on the bundle sha but left `run_projection`'s
`(tag, trait)`-only key (same #722-r3 stale-reuse class) one function below —
a resume-key fix round should grep the file for other `done = {...}` sets.
Sibling family: [[start-manifest-stale-artifact-done]] (stale
resume), [[count-gate-starved-by-resume-skip]] (resume starves a gate).

---
title: 'workflow-fix: non-canonical ''workflow-fix'' tag misses wf-fix floor predicate
  (61 tasks)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-22T11:28:37Z'
has_clean_result: false
origin_prompt: 'Found by the /issue 2291 orchestrator while checking whether the Step
  2 minimum plan-review floor bound to its own task: #2291 is kind:infra tagged ''workflow-fix''
  with no floor-triggering title prefix, so it misses both legs of the wf-fix-tag-OR-title-prefix
  predicate. 69 tasks carry the non-canonical tag; 61 of them miss both legs.'
workflow: v1
---
## Provenance

workflow_fix_target: src/explore_persona_space/task_workflow.py
fingerprint: noncanonical-workflow-fix-tag-misses-wf-fix-floor-predicate
Filed by: the `/issue 2291` orchestrator session, which hit this while checking
whether the Step-2 minimum plan-review floor bound to its own task. #2291 is
`kind: infra`, tagged `workflow-fix`, and its title carries no floor-triggering
prefix — so it misses the floor on BOTH legs. The floor was run anyway because
the orchestrator read the prose rather than trusting a predicate, so nothing was
lost on #2291 itself (same posture as #2282 → #2284).

## Goal

Canonicalize the workflow-fix task TAG so that workflow-fix tasks reliably
trigger the `kind: infra` workflow-fix predicates — principally the Step-2
minimum plan-review floor — instead of silently skipping them.

Two tag spellings are in live concurrent use for the same concept:

| tag | tasks carrying it |
|---|---|
| `wf-fix` (canonical — the value every predicate reads) | 1276 |
| `workflow-fix` (non-canonical) | 69 |

Measured 2026-08-22 by `grep -rlE '^- <tag>$' tasks/*/*/body.md`. The counts
drift upward during any session (they moved 1275→1276 / 68→69 within this one),
which is itself the point: this is not a historical artifact, it is an active
filing path.

## Why this matters (the concrete failure mode)

Per #2284, the floor's triggering class is the `wf-fix` TAG **OR** a title
prefix in `WF_FIX_TITLE_PREFIXES` (`workflow-fix:` / `daily-fix:`). A task
tagged `workflow-fix` fails the tag leg. If its title also lacks the prefix, it
fails both legs and the floor never binds.

**Of the 69 non-canonical-tag tasks, 61 miss BOTH legs.** The remaining 8 are
rescued only incidentally, by a title that happens to carry the prefix.

Live, not historical — the newest both-leg misses:

```
2026-08-22T02:54  #2461
2026-08-21T14:53  #2453
2026-08-21T04:32  #2440
2026-08-21T02:52  #2439
2026-08-21T00:55  #2438
2026-08-20T22:42  #2433
```

Beyond the plan-review floor, the same tag value gates the `file_infra_task.py`
backstops introduced by #1173 / #1283 / #1399 / #1502 (title-prefix WARN,
completed-sibling dedup screening, open-sibling advisory). A `workflow-fix`-tagged
filing is invisible to each of them, so the dedup and advisory machinery built
specifically to keep workflow-fix filings clean does not see 61 of them.

## Distinction from #2284 (do NOT dedupe onto it)

#2284 fixed a DOCUMENTATION defect: SKILL.md's floor paragraph stated a
tag-OR-title trigger and then attributed it to
`task_workflow.is_workflow_fix_session`, which actually tests only the
`workflow_fix_target:` provenance line. That is the predicate's *attribution*.

This task is a DATA/VOCABULARY defect: the predicate (post-#2284, correctly
described) reads the tag value `wf-fix`, and 69 tasks carry a different string
for the same concept. #2284's fix is necessary and does not close this — a
correctly-described tag predicate still cannot match a tag that was never
written in the canonical spelling.

## Scope

Left to the planner; the shape of the decision is:

1. **Where to normalize.** Accept `workflow-fix` as an alias at every read site
   (widen the predicates), or normalize at the WRITE site (`task.py new` /
   `add-tag` canonicalize on input), or both. Read-site aliasing fixes the 61
   existing tasks; write-site canonicalization stops new ones. They are not
   substitutes.
2. **Backfill or not.** Whether to retag the 69 existing tasks. Retagging
   mutates 69 task bodies at the shared repo root under concurrency — if taken,
   it needs the claim-marker + explicit-pathspec discipline from CLAUDE.md
   § Concurrent repo-root committers, not a bulk sweep.
3. **A mechanical guard** so the two spellings cannot re-diverge — the natural
   home is `workflow_lint.py`, alongside the existing wf-fix surface pins.

## Acceptance criteria

- Every currently-filed `kind: infra` workflow-fix task triggers the Step-2
  plan-review floor predicate, by tag or by an accepted alias — verified by
  re-running the both-leg-miss count above and getting 0.
- New filings cannot introduce a non-canonical spelling (write-site
  normalization, or a lint that fails on one).
- A regression pin covering the alias/normalization decision.
- No behavior change for the 1276 already-canonical tasks.

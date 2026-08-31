---
title: 'main-red: single bare hf_hub_download on main (issue1901_singleturn_retrieval_final.py:133)
  keeps --check-live-hf-retry-routing + no-flags red fleet-wide'
kind: infra
tags:
- main-red
created_at: '2026-08-31T05:16:31Z'
has_clean_result: false
parent_id: 2649
origin_prompt: 'Surfaced by #2649 post-merge acceptance on origin/main 2e3480c2df70:
  --check-live-hf-retry-routing FAIL(5); 4 errors are an untracked main-absent file
  (another session''s in-flight work), 1 is resident on main at scripts/issue1901_singleturn_retrieval_final.py:133,
  added by 94d6f8838f3 after #2649''s branch base. One-line fix; filed --no-dispatch
  to be batched with the other open main-red tasks rather than spawning a full cycle.'
workflow: v1
---
## Goal

Clear the one bare HF Hub call on `origin/main` that keeps
`workflow_lint.py --check-live-hf-retry-routing` (and therefore the no-flags
default run, and therefore `tests/test_workflow_lint.py::test_workflow_lint_default_exits_zero`)
red for every session's Step 9c gate.

## The single site

    scripts/issue1901_singleturn_retrieval_final.py:133   hf_hub_download(

Added to main by commit `94d6f8838f3` ("Add deduplicated single-turn retrieval
evaluation"). Measured on `origin/main` at `2e3480c2df70`:
`--check-live-hf-retry-routing` → `FAIL (5 error(s))`, of which this is the only
one resident on main (the other four are an untracked, main-absent
`scripts/issue2474_whiten_csls.py` belonging to a live session — see § Do not
conflate, below).

## Fix

The established one-line idiom (the same one #2649 applied at 10 sites):

```python
path = hub.retry_transient(lambda: hf_hub_download(...), what="<desc>")
```

`hub.retry_transient` is synchronous (`return fn()`), so wrapping an EAGER call
like `hf_hub_download` needs no waiver and no materialization trick. (Contrast
`list_repo_tree`, which is a lazy generator — there the consumption must be
materialized INSIDE the thunk; not applicable here.)

## Why this is filed rather than folded into #2649

#2649 (the main-red repair that just landed as PR #2129, merge commit
`942ad66d452`) enumerated 34 sites across 12 files and cleared all of them;
main is verified green on its whole scope. This site landed on main AFTER
#2649's branch base `59ebc5a6b27`, mid-round. #2649's approved plan makes
"editing outside the 12-file work table" an explicit must-ask boundary, so
absorbing a new arrival post-merge would have been a silent scope widening.
Filed instead, with the provenance recorded on #2649's `epm:merged`.

## Dispatch guidance (cost)

This is ONE line. Running a full `/issue` cycle for it — planner, critic
ensemble, code review, a ~40-min Step 10d gate — costs far more than the fix.
Filed with `--no-dispatch` deliberately: **batch it** with the other open
main-red / lint-routing tasks rather than spawning a dedicated session.
Related open siblings at filing time: #2586, #2648, #2650 (thread-caps /
torch-before-dotenv), #2573 + #2631 (Step 5a/10d sync-import attribution),
#2550 (the structural selector arm).

## The recurring-arrival problem this instance illustrates

Every newly-landed `scripts/*.py` carrying a bare Hub call re-reds main for the
whole fleet, so instance-fixing never converges — #2649 cleared 34 sites and a
new one arrived before it could finish merging. `tasks/REGISTRY.json` holds a
double-digit count of fix tasks in this class. The durable arm is a
land-time gate (pre-commit / CI on the adding commit) or #2550's
directory-membership selector proposal, not another repair task. Recording that
here so this task is understood as triage, not as a fix for the class.

## Acceptance criteria

1. `uv run python scripts/workflow_lint.py --check-live-hf-retry-routing` reports
   no error naming `scripts/issue1901_singleturn_retrieval_final.py` on a
   pristine `origin/main` worktree.
2. `uv run python scripts/workflow_lint.py` (no flags) reports no error naming
   that file.
3. `uv run pytest tests/test_workflow_lint.py` green (or, if red, every failure
   attributed to a cause other than this file, named).
4. No blanket grandfathering and no snapshot regen used to silence the check
   (`--regen-hf-routing-snapshot` is NOT the remedy here — the call is genuinely
   unrouted, not a stale-snapshot artifact).

## Do not conflate

`scripts/issue2474_whiten_csls.py` (4 further errors on the same check) is
**UNTRACKED** and **absent from `origin/main`** — a live session's in-flight
work that only appears in a root-side live-tree scan. It is that session's to
land correctly and is NOT in this task's scope. Do not file it, and do not
read a root-side `FAIL` as main being red on its account.

## Provenance

Surfaced by #2649's post-merge acceptance verification on `origin/main`
(2026-08-31 ~05:12Z), immediately after PR #2129 merged. Evidence:
`/tmp/i2649-acc2b.out`.

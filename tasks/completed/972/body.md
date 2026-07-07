---
title: 'workflow-fix: artifact-reuse check (i) also covers Hub-API call scoping'
kind: infra
tags:
- wf-fix
- wf-fix-fp:03a052106e6c
created_at: '2026-07-04T07:10:46Z'
has_clean_result: false
origin_prompt: 'orchestrator prose follow-up on #810 btdr round: reused helpers''
  unscoped list_repo_files full-tree verify wedged the live GPU run in 429 storms
  (gotchas #833 class); extend artifact-reuse fitness check (i) to Hub-API scoping'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #810 (emitting agent: orchestrator, /issue 810 btdr round).

## Goal

Extend the artifact-reuse fitness check (i) (throughput fitness of reused
fit/analysis code) to also cover Hub-API call scoping: reused code's
post-upload verify / staging calls must be prefix-scoped
(`list_repo_tree(path_in_repo=...)` / `file_exists`), never an unscoped
full-tree `list_repo_files` or `snapshot_download` against the ~1M-file data
repo.

## Workflow gap

- **Bug observed:** the #810 btdr follow-up round reused the parent
  extractor/upload helpers (`issue810_extract_positions.py` /
  `issue810_common.py`) whose post-upload verify ran an unscoped
  `list_repo_files` full-tree crawl of the data repo; live on 2026-07-04 it
  wedged in 429 retry storms (retry 20/20 reached) against the 2500-req/5-min
  org quota while an A100 idled at 0%.
- **Why it is a workflow gap:** `.claude/rules/artifact-reuse.md` check (i)
  mandates inspecting reused code's inner loop batching + device routing, but
  NOT its Hub-API call scoping — so a reused (not-touched-this-round) helper
  carrying the gotchas.md #833 anti-pattern passes plan review, the
  consistency-checker, and code review (which sees only the round's diff)
  un-inspected.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/rules/artifact-reuse.md`, fitness check (i):

```
- (i) throughput fitness of reused fit/analysis CODE — inner
-     per-cell/per-fold/per-draw loop batched + device parametrized
+ (i) throughput fitness of reused fit/analysis CODE — inner
+     per-cell/per-fold/per-draw loop batched + device parametrized, AND
+     Hub-API calls scoped: post-upload verifies / staging enumerate a
+     prefix (list_repo_tree path_in_repo= / file_exists), never a
+     full-tree list_repo_files / snapshot_download of the data repo
+     (gotchas.md #833; #810 btdr 429 wedge, 2026-07-04); a first-page
+     429 needs a bounded outer retry (the #658 rule)
```

Also consider a one-line pointer in `.claude/agents/code-reviewer.md` Step
0.68 (throughput checklist) so a NEW diff introducing the pattern is caught
too.

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`
- Secondary (planner's call): `.claude/agents/code-reviewer.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'list_repo_files' .claude/ CLAUDE.md`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/artifact-reuse.md
- fingerprint: 03a052106e6c

Surfaced prose (orchestrator observation, /issue 810 btdr round, 2026-07-04):
reused upload/verify helpers carried an unscoped full-tree list_repo_files
verify that wedged the live GPU run in 429 storms; hot-fixed in-flight as
04701b2d56; the reuse checklist should have surfaced it at plan time.

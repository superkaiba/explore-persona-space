---
title: 'workflow-fix: merge=union for eval_results/INDEX.md'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1af4b3f4a31b
- daily-auto-filed
created_at: '2026-07-19T07:06:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): eval_results/INDEX.md has
  merge: unspecified (verified via git check-attr) while the analogous append-only
  surfaces (tasks/**/events.jsonl, comments.jsonl, agent-memory) carry merge=union;
  the INDEX.md append-conflict class recurrently fires at sync_repo_root''s own pull
  (#1525).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose workflow-fix
follow-up raised on task #1525 (emitting agent: Alternatives plan-critic;
parked under the recursion guard, routed by the 2026-07-18 /daily Step C
parked-candidate sweep).

## Goal

Add `eval_results/INDEX.md merge=union` to `.gitattributes` (same
append-dominant rationale + LOCAL-merges-only scope caveat as the #896 block)
so the recurring INDEX.md append-conflict class never fires at
`sync_repo_root`'s own pull.

## Workflow gap

- **Bug observed:** `eval_results/INDEX.md` is append-dominant and repeatedly
  conflicts on concurrent repo-root pulls; #1525's stranded-commit incident
  included exactly this conflict class at sync_repo_root's own pull.
- **Why it is a workflow gap:** `.gitattributes` already carries
  `merge=union` for the analogous append-only surfaces
  (`tasks/**/events.jsonl`, `tasks/**/comments.jsonl`,
  `.claude/agent-memory/**/*.md`) and the #1525 session verified union IS
  honored on local `pull --rebase=merges` — INDEX.md is the missing sibling.
  Complements (does not replace) #1525's own recipe fix; the stranded-commit
  defect is path-general.
- **Confidence (emitter):** medium (the emitting critic pre-verified the
  attribute state and the union-honored-on-local-pull behavior)
- verified-at-filing: `git check-attr merge eval_results/INDEX.md` → `merge: unspecified` (gap present as claimed); `grep -n 'merge=union' .gitattributes` → 3 hits (L18 tasks/**/events.jsonl, L19 tasks/**/comments.jsonl, L30 .claude/agent-memory/**/*.md), no eval_results entry; `git log --oneline --since='7 days ago' -- .gitattributes` → 0 commits (no just-landed fix) (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up)

Sketch for the planner:

```
# .gitattributes, alongside the #896 append-only union block:
+ eval_results/INDEX.md merge=union
```

with the same scope-caveat comment as the existing block (union takes effect
ONLY on LOCAL merges/rebases, not GitHub-side merges).

## Scope / surfaces

- Primary target: `.gitattributes`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'merge=union\|INDEX.md' .claude/ CLAUDE.md scripts/ .gitattributes`)
  — check whether `scripts/sync_repo_root.py`'s byte-identical untracked-
  collision sweep or docs reference INDEX.md conflict handling; list every
  hit in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Union merge is only safe for append-dominant files — the plan states why
  INDEX.md qualifies (and what a mid-file edit would do under union).
- `scripts/workflow_lint.py --check-asks` passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: .gitattributes
- fingerprint: b0c00eab01cf

Verbatim surfaced prose (task #1525 events.jsonl, 2026-07-19T00:52:07Z):
"Candidate surfaced by the Alternatives plan-critic on #1525: add
'eval_results/INDEX.md merge=union' to .gitattributes (same append-dominant
rationale + scope caveat as the #896 block) so the recurring INDEX.md
append-conflict class never fires at sync_repo_root's own pull. Verified: git
check-attr merge eval_results/INDEX.md -> unspecified; .gitattributes already
carries merge=union for tasks/**/events.jsonl + comments.jsonl +
.claude/agent-memory/**/*.md and notes union IS honored on local pull
--rebase=merges. Complements (does not replace) this task's recipe fix — the
stranded-commit defect is path-general. target_file: .gitattributes."

---
title: 'workflow-fix: worktrees are a required location class in absence sweeps'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e0f976359360
created_at: '2026-08-05T05:14:52Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation during #1739 follow-up, 2026-08-05: three
  false absence claims in one session because the relocation sweep covered only the
  repo-root tree. The judged WildChat dv_dataset was untracked in .claude/worktrees/issue-1739;
  I told two subagents it did not exist and one nearly concluded it needed to re-judge
  10,000 rollouts. Neither absence-claim clause names worktrees (0 grep hits in both),
  while gotchas.md instructs excluding worktrees from recursive greps -- so following
  both rules correctly produces a sweep blind to where active in-flight work lives.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a candidate raised during
task #1739 follow-up work (emitting agent: orchestrator, own observation).

## Goal

Name WORKTREES as a required location class in the absence-claim relocation
sweep, and resolve the conflict with the `gotchas.md` rule that instructs
excluding worktrees from recursive greps.

## Workflow gap

- **Bug observed:** three absence claims in one session were false because the
  relocation sweep covered only the repo-root tree. The artifacts were untracked
  inside `.claude/worktrees/issue-1739/`. The most costly instance: I told two
  subagents that `eval_results/issue_1739/wildchat_rung/dv_dataset/<behavior>/labeling.json`
  "DOES NOT EXIST" and instructed them to sweep for it; one was a step away from
  concluding it needed to re-judge 10,000 rollouts. The files were present, 2,000
  rows per behaviour, in the worktree.
- **Why it is a workflow gap:** the two absence-claim clauses are explicit about
  WHAT to sweep but silent on worktrees as a location class, and a sibling rule
  actively steers agents away from them. CLAUDE.md's "Absence claims require a
  relocation sweep" specifies "repo-wide grep plus a SCOPED listing of the HF
  prefix's siblings/parents"; `.claude/rules/workflow-fix-on-bug.md` clause (b)
  specifies `grep -rn '<symbol>' tests/ scripts/ .claude/ src/`. Neither names
  worktrees. Meanwhile `gotchas.md` says: "Recursive greps spanning `.claude/`
  traverse the worktrees bind-mount ... and hang/time out. RULE: exclude the
  mount (`grep -rn --exclude-dir=worktrees ...`)". So an agent following BOTH
  rules correctly performs a sweep that systematically cannot see worktree
  contents — and on this VM every active `/issue` session works in a worktree,
  so that is precisely where in-flight, not-yet-committed artifacts live. The
  gotchas exclusion is CORRECT for its own purpose (grepping `.claude/` config
  for an idiom); it is wrong when the question is "does this artifact exist
  anywhere". The rules do not currently distinguish those two cases.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c 'worktree'` against the extracted "Absence claims require a relocation sweep" clause in `CLAUDE.md` → **0 hits**; `sed -n '486,496p' .claude/rules/workflow-fix-on-bug.md | grep -c 'worktree'` (clause (b), relocation grep) → **0 hits**; conflicting guidance present and quoted from `.claude/rules/gotchas.md` (the "Recursive greps spanning `.claude/`" entry, #1773). Absence-of-guard claim, so the 0-hit in-target result IS the evidence (§ verified-at-filing clause (a)). Landed-fix check: `git log --oneline --since='7 days ago' -- CLAUDE.md` reviewed; no commit addresses worktree coverage in the absence-sweep clauses. (2026-08-05)

## Proposed change (candidate diff sketch — refine in planning)

In CLAUDE.md's "Absence claims require a relocation sweep" clause, and the
parallel clause (b) in `.claude/rules/workflow-fix-on-bug.md`:

```
  **Absence claims require a relocation sweep:** ... grounded by a relocation
  sweep — repo-wide grep plus a SCOPED listing of the HF prefix's
- siblings/parents ... never a single-location check
+ siblings/parents ..., PLUS the WORKTREES (`.claude/worktrees/*/`), never a
+ single-location check. Worktrees are a REQUIRED location class, not an
+ optional extra: every active /issue session works in one, so in-flight and
+ not-yet-committed artifacts live there by default, and an untracked file in
+ a worktree is invisible to `git ls-files`, to a repo-root `find`, and to a
+ `--exclude-dir=worktrees` grep. Sweep them with a BOUNDED probe — a targeted
+ `ls`/`find` on the specific expected path under each `.claude/worktrees/*/`,
+ or `git -C <wt> status --porcelain` — NEVER an unbounded `grep -r`, which
+ traverses the multi-GB bind-mounted data caches and hangs (the gotchas.md
+ `--exclude-dir=worktrees` rule). The exclusion rule governs CONFIG greps of
+ `.claude/`; it does not license omitting worktrees from an EXISTENCE check.
```

Planning should decide whether this also belongs in the teammate-coordination
durable-state probe (CLAUDE.md clause (b) of the teammate bullet), where the
same blind spot let an untracked 1,190-line in-progress script read as "zero
durable state" — though note that clause already says to run `git status` on
"its worktree AND the repo root", so that may be a compliance issue rather than
a gap and should not be widened reflexively.

## Scope / surfaces

- Primary target: `CLAUDE.md`
- Sibling: `.claude/rules/workflow-fix-on-bug.md` clause (b)
- Grep the surface before editing (`grep -rn --exclude-dir=worktrees 'relocation sweep\|relocation grep' CLAUDE.md .claude/`) and keep the new text consistent with the gotchas.md exclusion rule it must explicitly reconcile rather than contradict.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Must NOT tell agents to run unbounded recursive greps over worktrees; the
  bounded-probe form is the whole point of the reconciliation.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: CLAUDE.md
- fingerprint: e0f976359360

<!-- workflow-fix-candidate v1 -->
target_file: CLAUDE.md
bug_observed: three absence claims in one session were false because the relocation sweep covered only the repo-root tree; the artifacts were untracked inside .claude/worktrees/issue-1739, and one false claim nearly triggered a re-judge of 10000 rollouts
why_workflow_gap: neither absence-claim clause names worktrees as a location class (0 grep hits in both), while gotchas.md instructs excluding worktrees from recursive greps for performance — so an agent following both rules correctly runs a sweep that cannot see the place active in-flight artifacts actually live
proposed_change: name worktrees as a required location class in the absence-claim relocation sweep, resolving the conflict with the gotchas rule that excludes worktrees from recursive greps
diff_sketch: |
  + PLUS the WORKTREES (`.claude/worktrees/*/`) — a REQUIRED location class:
  + every active /issue session works in one, so in-flight untracked artifacts
  + live there and are invisible to git ls-files, a repo-root find, and a
  + --exclude-dir=worktrees grep. Sweep with a BOUNDED probe (targeted ls/find
  + on the expected path, or `git -C <wt> status --porcelain`), never an
  + unbounded grep -r, which hangs on the bind-mounted data caches.
confidence: high
related_task: #1739
<!-- /workflow-fix-candidate -->

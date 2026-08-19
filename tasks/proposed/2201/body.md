---
title: 'workflow-fix: pre-review deliverable-divergence probe (Step 5a/10d)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:55342a328f78
created_at: '2026-08-08T16:03:13Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: .claude/skills/issue/SKILL.md\n\
  bug_observed: The #1771 branch reached round-1 code review with origin/main having\
  \ rewritten the same function + test file in the opposite direction (#2164, landed\
  \ 2026-08-07); nothing in Step 5a spec-freshness sync or the review dispatch flagged\
  \ that 16/20 round-deliverable files had diverged on main since the merge-base.\n\
  why_workflow_gap: Step 5a syncs never-branch-edited sibling files but is silent\
  \ about deliverable files main has since modified, so a semantic collision (open-sibling\
  \ #1479/#1476 class, now at review/merge grain) surfaces only as a raw rebase conflict\
  \ at Step 10d with no implementer in the loop.\nproposed_change: Add a cheap pre-review/pre-merge\
  \ divergence probe that lists round-deliverable files also changed on origin/main\
  \ since the merge-base and injects the list into the code-review brief (and blocks\
  \ Step 10d auto-merge when the intersection is non-empty, routing to an implementer\
  \ reconciliation round).\ndiff_sketch: |\n  + Step 5a (after sync): MB=$(git merge-base\
  \ origin/main HEAD);\n  + comm -12 <(git diff --name-only $MB..HEAD | sort) <(git\
  \ diff --name-only $MB origin/main | sort)\n  + -> non-empty => append a \"DIVERGED-ON-MAIN\
  \ deliverables\" list to the reviewer brief;\n  + Step 10d: same probe; non-empty\
  \ => do NOT auto-resolve — dispatch an implementer\n  + reconciliation round naming\
  \ the diverged files.\nconfidence: medium\nrelated_task: #1771\n<!-- /workflow-fix-candidate\
  \ -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1771 (emitting agent: code-reviewer, round 1, 2026-08-08).

## Goal

Add a cheap pre-review/pre-merge divergence probe that lists round-deliverable
files also changed on origin/main since the merge-base and injects the list
into the code-review brief (and blocks Step 10d auto-merge when the
intersection is non-empty, routing to an implementer reconciliation round).

## Workflow gap

- **Bug observed:** The #1771 branch reached round-1 code review with
  origin/main having rewritten the same function + test file in the opposite
  direction (#2164, landed 2026-08-07 via merge 5c91482fce); nothing in
  Step 5a spec-freshness sync or the review dispatch flagged that
  round-deliverable files had diverged on main since the merge-base.
  (The "16/20 files diverged" figure is reviewer-measured — see the
  `epm:code-review v1` marker on #1771, 2026-08-08T15:58:24Z, and
  /tmp/issue-1771-code-review-r1.md; re-measure at plan time.)
- **Why it is a workflow gap:** Step 5a syncs never-branch-edited sibling
  files but is silent about deliverable files main has since modified, so a
  semantic collision (the open-sibling #1479/#1476 class, now at review/merge
  grain) surfaces only as a raw rebase conflict at Step 10d with no
  implementer in the loop.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "DIVERGED\|diverged on main\|divergence probe" .claude/skills/issue/SKILL.md` → 3 hits (repo-root main copy, 2026-08-08), each read in context: lines 2080/13149 are the #1725 ROOT-divergence probe (local `main` vs `origin/main` at the shared repo root — a different object than branch deliverables vs main) and line 10932 an unrelated sync-site list; Step 10d's merge-conflict handling (102 "conflict" mentions) is REACTIVE recovery at merge time, not a pre-review flag. Absence-of-guard claim: 0 hits for a pre-review deliverable-divergence probe.

## Proposed change (candidate diff sketch — refine in planning)

```
+ Step 5a (after sync): MB=$(git merge-base origin/main HEAD);
+ comm -12 <(git diff --name-only $MB..HEAD | sort) <(git diff --name-only $MB origin/main | sort)
+ -> non-empty => append a "DIVERGED-ON-MAIN deliverables" list to the reviewer brief;
+ Step 10d: same probe; non-empty => do NOT auto-resolve — dispatch an implementer
+ reconciliation round naming the diverged files.
```

Planning notes: (a) exclude spec-freshness sync commits' paths from the
branch side (subject-scoped exclusion, as Step 5a already does) or the probe
false-positives on every synced file; (b) the Step 10d leg should compose
with — not duplicate — the existing Known-failure-shape merge-conflict
recovery (reactive); the new value is the PROACTIVE pre-review flag and the
implementer-in-the-loop reconciliation routing; (c) unverified hypothesis —
verify at plan time: whether a same-file-touched-on-both-sides list is
low-noise enough on hot files (CLAUDE.md, SKILL.md are touched on main
near-daily) or needs a hunk-overlap / same-symbol refinement to avoid
flagging every round.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'spec-freshness' .claude/ CLAUDE.md scripts/`) and update every
  coupled surface the probe touches; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 55342a328f78

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: The #1771 branch reached round-1 code review with origin/main having rewritten the same function + test file in the opposite direction (#2164, landed 2026-08-07); nothing in Step 5a spec-freshness sync or the review dispatch flagged that 16/20 round-deliverable files had diverged on main since the merge-base.
why_workflow_gap: Step 5a syncs never-branch-edited sibling files but is silent about deliverable files main has since modified, so a semantic collision (open-sibling #1479/#1476 class, now at review/merge grain) surfaces only as a raw rebase conflict at Step 10d with no implementer in the loop.
proposed_change: Add a cheap pre-review/pre-merge divergence probe that lists round-deliverable files also changed on origin/main since the merge-base and injects the list into the code-review brief (and blocks Step 10d auto-merge when the intersection is non-empty, routing to an implementer reconciliation round).
diff_sketch: |
  + Step 5a (after sync): MB=$(git merge-base origin/main HEAD);
  + comm -12 <(git diff --name-only $MB..HEAD | sort) <(git diff --name-only $MB origin/main | sort)
  + -> non-empty => append a "DIVERGED-ON-MAIN deliverables" list to the reviewer brief;
  + Step 10d: same probe; non-empty => do NOT auto-resolve — dispatch an implementer
  + reconciliation round naming the diverged files.
confidence: medium
related_task: #1771
<!-- /workflow-fix-candidate -->

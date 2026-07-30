---
title: 'daily-fix: Guard-4 grep -Fxq needs -- (false LOST-UPDATE)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3323474ad522
- daily-auto-filed
created_at: '2026-07-29T07:05:49Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): Step 10d Guard 4 (lost-update
  refusal, #1713) fenced block runs `grep -Fxq "$ADD_LINE"` without `--` (SKILL.md:10425);
  any main-added line beginning with `-` (markdown bullets — ubiquitous on the workflow
  surface) is parsed as a grep option, grep errors ("invalid option"), the non-zero
  rc counts the line as MISSING, and Guard 4 false-refuses the merge (reproduced live
  on #1742: 3 grep "invalid opt'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step C parked-candidate sweep (2026-07-28) from TWO independently emitted, same-bug parked candidates: task #1742 (ts 2026-07-28T09:37:55Z, fp 3323474ad522) and task #1758 (ts 2026-07-28T09:00:25Z, fp 2c4af412ddce). Both sessions hit the bug LIVE. A group-E transcript miner independently re-observed the same class the same day (one firing in session 24c4df99).

## Goal

Add the `--` end-of-options separator to the Step 10d Guard 4 (lost-update refusal) membership grep so main-added lines beginning with `-` (markdown bullets) are matched as patterns, not parsed as grep options.

## Workflow gap

- **Bug observed:** Guard 4's fenced recipe runs `grep -Fxq "$ADD_LINE"` without `--`; any main-added line starting with `-` makes grep error "invalid option", the non-zero rc counts the line as MISSING_ON_BRANCH, and the guard false-fires a LOST-UPDATE REFUSAL on a branch whose file is byte-identical to origin/main. Reproduced live twice on 2026-07-28: #1742 (3 phantom misses naming .claude/rules/planner-section-reference.md) and #1758 (3 phantom misses, clean rerun after adding `--`).
- **Why it is a workflow gap:** the guard's own fence is quoting-unsafe for the exact content class it scans (markdown workflow-surface files, where dash-leading bullets are ubiquitous); every Step 10d merge copy-runs the recipe verbatim, so false refusals block legitimate merges.
- **Confidence (emitter):** high (both emitters)
- verified-at-filing: `grep -rn --exclude-dir=worktrees 'grep -Fxq "$ADD_LINE"' .claude/ CLAUDE.md scripts/` → 1 hit in 1 file (.claude/skills/issue/SKILL.md:10510 — the Guard-4 block); the fixed form `grep -Fxq -- "$ADD_LINE"` has 0 hits in that file (2026-07-29 UTC). Landed-fix history check: `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` shows no commit fixing this grep (the #1753 Guard-4 recovery-ordering commit 9f5b75b4f3 touches Guard 4 but the unfixed grep is still live at 10510).

## Proposed change (candidate diff sketch — refine in planning)

```diff
-               | grep -Fxq "$ADD_LINE"; then
+               | grep -Fxq -- "$ADD_LINE"; then
```

Optionally materialize the branch copy to a temp file and use the file-arg form, as the #1742 session's clean rerun did.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing (`grep -rn 'grep -Fxq' .claude/ CLAUDE.md scripts/`) and update every unfixed hit; list them in the plan. A broader `grep -n 'grep -Fxq' .claude/skills/issue/SKILL.md` at filing time found THREE dash-unsafe sites (lines 10375, 10484, 10510) — fix all in lockstep. Check whether any pin test asserts the Guard-4 block text and update it in the same commit.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 3323474ad522

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: Step 10d Guard 4 (lost-update refusal, #1713) fenced block runs `grep -Fxq "$ADD_LINE"` without `--` (SKILL.md:10425); any main-added line beginning with `-` (markdown bullets — ubiquitous on the workflow surface) is parsed as a grep option, grep errors ("invalid option"), the non-zero rc counts the line as MISSING, and Guard 4 false-refuses the merge (reproduced live on #1742: 3 grep "invalid option" errors + a spurious LOST-UPDATE REFUSAL naming .claude/rules/planner-section-reference.md(3); clean rerun after adding --).
why_workflow_gap: the guard's own fence is quoting-unsafe for the exact content class it scans (markdown workflow-surface files), producing false merge refusals on every branch whose synced files contain dash-leading main-added lines.
proposed_change: add `--` end-of-options to the Guard-4 membership grep (optionally materialize the branch copy to a temp file and use the file-arg form, as the #1742 session's clean rerun did).
confidence: high
related_task: #1742
<!-- /workflow-fix-candidate -->

(Second same-bug park, task #1758, fp 2c4af412ddce, ts 2026-07-28T09:00:25Z — deduped into this filing; its routed-record on #1758 names this task.)

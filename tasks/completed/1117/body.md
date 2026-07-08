---
title: 'workflow-fix: hard-FAIL tier >=100 words/bullet in check_v4_word_caps'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b906ff111c7f
created_at: '2026-07-08T00:27:59Z'
has_clean_result: false
origin_prompt: 'clean-result-critic prose candidate on #825 r1 (2026-07-08): check_v4_word_caps
  WARNs identically on a 35-word and a 263-word Takeaways bullet; add a >=100-words/bullet
  hard-FAIL tier'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #825 (emitting agent: clean-result-critic).

## Goal

Add a second hard-FAIL tier at >=100 words per Takeaways bullet in `check_v4_word_caps` so an accreted paragraph-bullet cannot ride a WARN.

## Workflow gap

- **Bug observed:** check_v4_word_caps WARNs identically on a 35-word and a 263-word Takeaways bullet.
- **Why it is a workflow gap:** the v4 spec caps Takeaways bullets (~30 words WARN), but a same-issue follow-up re-fold can accrete a 263-word paragraph-bullet that still ships as a WARN — the mechanical verifier cannot distinguish mild overrun from accretion, so the clean-result-critic burns a REVISE round on what should be a mechanical FAIL.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ In scripts/verify_task_body.py check_v4_word_caps: keep the ~30-word WARN tier; add a hard-FAIL tier when any ## Takeaways bullet >= 100 words.
+ Mirror the two-tier wording in .claude/skills/clean-results/SPEC.md (conciseness caps section).
+ Grandfathering: v4 bodies only; v3/v2/legacy untouched (forward-only rule).

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`, `.claude/skills/clean-results/SPEC.md`
- Grep the workflow surface for the word-cap wording before editing (`grep -rln 'word cap\|check_v4_word_caps\|check 20' .claude/ scripts/verify_task_body.py`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; SPEC.md and the verifier stay consistent.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py, .claude/skills/clean-results/SPEC.md
- fingerprint: b906ff111c7f

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py, .claude/skills/clean-results/SPEC.md
bug_observed: check_v4_word_caps WARNs identically on a 35-word and a 263-word Takeaways bullet
why_workflow_gap: the verifier cannot distinguish mild overrun from accretion, so a 263-word paragraph-bullet ships as a WARN and costs a clean-result-critic REVISE round
proposed_change: add a second hard-FAIL tier at >=100 words per Takeaways bullet in check_v4_word_caps
diff_sketch: |
  + if bullet_words >= 100: FAIL (v4 bodies only)
  + keep ~30-word WARN tier; mirror in SPEC.md conciseness caps
confidence: medium
related_task: #825
<!-- /workflow-fix-candidate -->

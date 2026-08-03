---
title: 'daily-fix: register width-reeval pointer test in WORKFLOW_IN'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4cd5bd788c26
- daily-auto-filed
created_at: '2026-07-23T06:38:40Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): tests/test_issue_skill_width_reeval_pointer.py
  pins SKILL.md Step 6d.2 width-re-evaluation prose but is absent from WORKFLOW_INVARIANT,
  so a SKILL.md-only diff dropping the clause is never gated (rules-pin arm covers
  only the rule-file legs)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-22 parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1594 (emitting agent: planner, §2 prior-work note).

## Goal

Register `tests/test_issue_skill_width_reeval_pointer.py` in `WORKFLOW_INVARIANT` in `scripts/select_step9c_tests.py` (or otherwise close the SKILL.md-diff selection leg) so a Step-9c gate over a diff touching the pinned SKILL.md prose actually executes the pin test.

## Workflow gap

- **Bug observed:** `tests/test_issue_skill_width_reeval_pointer.py` pins the #1346 width-re-evaluation prose across FOUR surfaces including `.claude/skills/issue/SKILL.md` Step 6d.2 (`test_eta_advisory_names_width_reevaluation` reads SKILL.md directly), but it is absent from the `WORKFLOW_INVARIANT` selection, so a Step-9c gate over a diff that touches ONLY SKILL.md (dropping the width clause) may never execute the test. verify_plan c41 WARN surfaced the same gap on #1594.
- **Why it is a workflow gap:** the sibling SKILL.md-pin tests (`test_issue_skill_exit_breadcrumb.py`, the #1575/#1546/#1268/#1563/#1595/#1587 pins, `test_step10d_guard3.py`) ARE registered in WORKFLOW_INVARIANT (select_step9c_tests.py lines ~270-289); this one was left out.
- **Confidence (emitter):** low-medium (emitter); raised to medium-high at filing after the semantic probe below.
- verified-at-filing: `grep -c 'test_issue_skill_width_reeval_pointer' scripts/select_step9c_tests.py` → 0 hits in the named target (absence-of-registration claim — the 0-hit IS the evidence), 2026-07-23 UTC. Semantic probe of the emitter's own "or confirm the #1496 arm maps it" escape: `rules_pin_pairs(['.claude/rules/crash-fix-rounds.md','.claude/rules/vectorize-many-cell-fits.md'], Path('.'))` RETURNS the test for both rule files — so the RULE-file diff legs are covered — but the test ALSO pins `.claude/skills/issue/SKILL.md` Step 6d.2 (read tests/test_issue_skill_width_reeval_pointer.py:14-22), and rules_pin_pairs keys on the rule files only, so a SKILL.md-only diff does not select it. Landed-fix history check: `git log --oneline --since='7 days ago' -- scripts/select_step9c_tests.py` → 5 commits (4c4e79688b, 26a89e93bb, 0e5d05ba95, 7647f3fb34, c010cffc20), none registering this test. Note: #1593 (c010cffc20) replaced the WORKFLOW_INVARIANT count pin with a sorted-manifest set-equality pin — adding a member requires updating that manifest too.
- Open-sibling note: open wf-fix #865 (on_hold) targets the same file but a DIFFERENT bug (selector diffs main checkout, blind to worktree branches) — not a duplicate under the (target_file, fingerprint) grain.

## Proposed change (candidate diff sketch — refine in planning)

Add `"tests/test_issue_skill_width_reeval_pointer.py"` to `WORKFLOW_INVARIANT` in `scripts/select_step9c_tests.py` with a `# NEW (#1346/#1594) — SKILL.md width-re-evaluation pin` comment, and update the sorted-manifest set-equality pin (#1593) accordingly. Alternative if the planner prefers: extend the transitive-consumer / rules-pin map so SKILL.md-pin tests are selected on SKILL.md diffs generically.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Also touch: the #1593 sorted-manifest set-equality pin (wherever the manifest lives — likely tests/) when adding a member.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; the Step-9c selector's own tests pass.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 4cd5bd788c26

- workflow_fix_target: scripts/select_step9c_tests.py

Verbatim parked candidate (prose park, task #1594 events 2026-07-22T13:22:08Z):

> parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see workflow-fix-on-bug § Recursion guard. source: prose-followup (planner §2 prior-work note). target_file: scripts/select_step9c_tests.py. bug_observed: tests/test_issue_skill_width_reeval_pointer.py pins crash-fix-rounds.md + vectorize-many-cell-fits.md prose but is absent from the WORKFLOW_INVARIANT selection, so a Step-9c gate over a diff touching those rules may not execute it (verify_plan c41 WARN surfaced the same gap on this task). proposed_change: register it in WORKFLOW_INVARIANT (or confirm the #1496 rules-pin dependency arm already maps it — this round's Step-10d TG map n=45 may have included it, which would narrow the gap to the Step-9c leg only). confidence: low-medium. related_task: #1594.

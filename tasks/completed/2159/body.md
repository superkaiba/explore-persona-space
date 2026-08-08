---
title: 'workflow-fix: Repoint the pin''s read target from CLAUDE.md to .claude/rule'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0044a9d9019b
- urgent-main-red
created_at: '2026-08-07T00:13:20Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: tests/test_issue_skill_trigger_dense_tag_adoption.py\n\
  bug_observed: test_goal_channel_clause_present fails on origin/main — it greps CLAUDE.md\
  \ for the refusal-ladder clause \"(e) prevention beats recovery\" + \"pass the Goal\
  \ BY REFERENCE\", but the 2026-08-05/06 compaction moved the full refusal ladder\
  \ out of CLAUDE.md into .claude/rules/context-hygiene.md (both strings verified\
  \ present there).\nwhy_workflow_gap: The compaction train landed without re-running\
  \ the workflow-invariant pin family on the final merged tree, leaving a stale pin\
  \ target that every intervening Step 9c gate must re-classify as known-red.\nproposed_change:\
  \ Repoint the pin's read target from CLAUDE.md to .claude/rules/context-hygiene.md\
  \ (content is preserved verbatim there); keep the window assertions unchanged.\n\
  diff_sketch: |\n  - CLAUDE_MD = REPO / \"CLAUDE.md\"\n  + CONTEXT_HYGIENE_MD = REPO\
  \ / \".claude\" / \"rules\" / \"context-hygiene.md\"\n    def test_goal_channel_clause_present():\n\
  \  -     text = CLAUDE_MD.read_text(encoding=\"utf-8\")\n  +     text = CONTEXT_HYGIENE_MD.read_text(encoding=\"\
  utf-8\")\n        i = text.index(\"(e) prevention beats recovery\")\nconfidence:\
  \ high\nrelated_task: #2157\nurgency: main-red\nfailing_test: tests/test_issue_skill_trigger_dense_tag_adoption.py::test_goal_channel_clause_present\n\
  wf_fix: true\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#2157. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_issue_skill_trigger_dense_tag_adoption.py::test_goal_channel_clause_present` red on origin/main: Repoint the pin's read target from CLAUDE.md to .claude/rules/context-hygiene.md (content is preserved verbatim there); keep the window assertions unchanged.

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** test_goal_channel_clause_present fails on origin/main — it greps CLAUDE.md for the refusal-ladder clause "(e) prevention beats recovery" + "pass the Goal BY REFERENCE", but the 2026-08-05/06 compaction moved the full refusal ladder out of CLAUDE.md into .claude/rules/context-hygiene.md (both strings verified present there).
- **Failing node (router-verified):** `tests/test_issue_skill_trigger_dense_tag_adoption.py::test_goal_channel_clause_present`
- **Confidence (emitter):** high
- verified-at-filing: `uv run pytest tests/test_issue_skill_trigger_dense_tag_adoption.py::test_goal_channel_clause_present -q` -> rc=1 at main @ 1db0077c1a (2026-08-07T00:13:08Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `tests/test_issue_skill_trigger_dense_tag_adoption.py`
- Failing node: `tests/test_issue_skill_trigger_dense_tag_adoption.py::test_goal_channel_clause_present`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- This session carries a `workflow_fix_target:` Provenance line — it MUST
  NOT auto-route its own subagents' workflow-fix candidates (recursion
  guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: tests/test_issue_skill_trigger_dense_tag_adoption.py
- fingerprint: 0044a9d9019b
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: tests/test_issue_skill_trigger_dense_tag_adoption.py
bug_observed: test_goal_channel_clause_present fails on origin/main — it greps CLAUDE.md for the refusal-ladder clause "(e) prevention beats recovery" + "pass the Goal BY REFERENCE", but the 2026-08-05/06 compaction moved the full refusal ladder out of CLAUDE.md into .claude/rules/context-hygiene.md (both strings verified present there).
why_workflow_gap: The compaction train landed without re-running the workflow-invariant pin family on the final merged tree, leaving a stale pin target that every intervening Step 9c gate must re-classify as known-red.
proposed_change: Repoint the pin's read target from CLAUDE.md to .claude/rules/context-hygiene.md (content is preserved verbatim there); keep the window assertions unchanged.
diff_sketch: |
  - CLAUDE_MD = REPO / "CLAUDE.md"
  + CONTEXT_HYGIENE_MD = REPO / ".claude" / "rules" / "context-hygiene.md"
    def test_goal_channel_clause_present():
  -     text = CLAUDE_MD.read_text(encoding="utf-8")
  +     text = CONTEXT_HYGIENE_MD.read_text(encoding="utf-8")
        i = text.index("(e) prevention beats recovery")
confidence: high
related_task: #2157
urgency: main-red
failing_test: tests/test_issue_skill_trigger_dense_tag_adoption.py::test_goal_channel_clause_present
wf_fix: true
<!-- /workflow-fix-candidate -->

---
title: 'workflow-fix: retro-close predicate ignores queued-context label mentions'
kind: infra
tags:
- wf-fix
created_at: '2026-07-04T04:30:44Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised by /issue 825 orchestrator 2026-07-04:
  followup_retro_close_evidence false positive on queued-label park notes (see body
  Provenance block)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #825 (emitting agent: orchestrator, /issue 825 session).

## Goal

Reject status/step-note retro-close evidence when the label token appears in a queued/unrun context, or require the round-completion word to bind to the label's clause, in followup_retro_close_evidence; add a regression test replaying the #825 note.

## Workflow gap

- **Bug observed:** followup_retro_close_evidence returned retro-close evidence for a genuinely UNRUN label: a step-completed note named the label only in a queued context ('1 unrun user-chat label (role-map-comparison) queued') while its round-completion word referred to a different round (#825, 2026-07-04T04:21:23Z).
- **Why it is a workflow gap:** the predicate's status/step-note evidence class matches any note containing the exact parenthesized label token plus a round-completion word ANYWHERE in the note, without binding the completion word to the label's clause and without excluding queue-context mentions ("queued", "unrun", "next entry dispatches"). Park/handoff notes routinely enumerate the queued next label while announcing ANOTHER round's completion, so every such park note is false retro-close evidence. Acting on it would retroactively close a queued USER-initiated round (dropping the user's ask) — only the orchestrator's manual cross-check against the handoff markers caught it on #825.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
In followup_retro_close_evidence (src/explore_persona_space/task_workflow.py),
status/step-note evidence class:
+ QUEUE_CONTEXT = re.compile(
+     r"(?:unrun|queued|next\s+entry\s+dispatches)[^.\n](0, 80)\(" + re.escape(label) + r"\)"
+     r"|\(" + re.escape(label) + r"\)[^.\n](0, 80)(?:queued|unrun)", re.I)
+ if QUEUE_CONTEXT.search(note):
+     continue   # label mentioned as QUEUED, not completed — not evidence
(alternative: require the completion word within the same sentence/clause as
the label token). Add regression test to tests/test_workflow_followup_labels.py
replaying the #825 2026-07-04T04:21:23Z step-completed note verbatim
(expect None) alongside the existing positive fixtures.
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Tests: `tests/test_workflow_followup_labels.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'followup_retro_close_evidence' src/ scripts/ tests/ .claude/skills .claude/agents`)
  and update every canonical (non-worktree) hit; the SKILL.md Stale-label
  prose may need a one-line note that queue-context mentions are excluded.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Existing positive retro-close fixtures must keep passing (the fix narrows
  false positives only).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py, tests/test_workflow_followup_labels.py
- fingerprint: 27b56e7fed52

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py, tests/test_workflow_followup_labels.py
bug_observed: followup_retro_close_evidence returned retro-close evidence for a genuinely UNRUN label: a step-completed note named the label only in a queued context ('1 unrun user-chat label (role-map-comparison) queued') while its round-completion word referred to a different round (#825, 2026-07-04T04:21:23Z)
why_workflow_gap: status/step-note evidence class matches '(label)' + completion word anywhere in the note, so park notes enumerating the queued label while announcing another round's completion are false evidence
proposed_change: Reject status/step-note retro-close evidence when the label token appears in a queued/unrun context, or require the round-completion word to bind to the label's clause, in followup_retro_close_evidence; add a regression test replaying the #825 note
diff_sketch: |
  exclude queue-context label mentions (unrun|queued|next entry dispatches near '(label)')
  from the status/step-note evidence class; regression test replaying the #825 note
confidence: high
related_task: #825
<!-- /workflow-fix-candidate -->

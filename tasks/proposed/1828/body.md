---
title: 'workflow-fix: harden _breadcrumb_fields against prose-glued/empty key=value
  tokens'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2331b34907b3
created_at: '2026-07-29T13:02:47Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1689 (2026-07-29): _breadcrumb_fields
  parses ''label=wider-lambda-ceilings:'' (glued colon, event v110) as a non-matching
  label and a prose ''label='' substring (event v115) as an empty re-bind; executing_followup_label
  then falls back to the queue-head unrun label, misidentifying the executing round
  for the watcher run-marker post / step-4 completion marker / tick_triage pid probe.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1689 (emitting agent: issue-orchestrator).

## Goal

Harden `_breadcrumb_fields` token parsing so prose-adjacent tokens (glued
trailing punctuation, bare `key=` substrings in prose) cannot misroute
`executing_followup_label` to the queue-head fallback.

## Workflow gap

- **Bug observed:** stage-dispatch breadcrumb label token with glued trailing
  punctuation (`label=wider-lambda-ceilings:`) or a prose `label=` substring
  re-binding the field to empty makes `executing_followup_label` fall back to
  the queue-head unrun label, misidentifying the executing round.
- **Why it is a workflow gap:** `_breadcrumb_fields` whitespace-splits
  `key=value` tokens with last-wins semantics and no value hygiene, so (a) a
  breadcrumb author who ends the token run with a prose colon
  (`label=wider-lambda-ceilings: detached ...`, task #1689 event v110,
  2026-07-29T08:28:32Z) produces a label that matches no group, and (b) any
  prose containing a bare `key=` sequence (`"its label= token was
  prose-glued"`, #1689 event v115) re-binds the field to EMPTY, silently
  discarding the correct earlier token. Both make
  `executing_followup_label` fall back to the DISPATCHABLE queue head — on
  #1689 that was the user-chat label `derived-vs-free-answer-map`, NOT the
  executing round `wider-lambda-ceilings`. Downstream consumers of the
  helper (`autonomous_session_watch._post_followup_run_marker`'s on-behalf
  run-marker post; the Step 9b step-4 completion-marker label derivation;
  `tick_triage.py`'s pid-probe crumb read) would then close / probe the
  WRONG label — a never-run user-chat round would be marked run.
- **Confidence (emitter):** high (live repro on #1689 events v110/v115/v116,
  2026-07-29: v110 parsed `label='wider-lambda-ceilings:'` → no group match;
  v115 parsed `label=''` via prose re-bind; the equals-free v116 re-post
  resolved correctly).
- verified-at-filing: `grep -n "def _breadcrumb_fields" src/explore_persona_space/task_workflow.py` → 1 hit (line 1907) in the named target; consumers grep `grep -rln "_breadcrumb_fields" src/ scripts/` → `src/explore_persona_space/task_workflow.py` + `scripts/tick_triage.py` (consumer via the shared helper, no independent parser). Live-repro parse values reproduced at filing time via `task_workflow._breadcrumb_fields` on #1689 events v110 (`'wider-lambda-ceilings:'`) and v115 (`''`) (2026-07-29).

## Proposed change (candidate diff sketch — refine in planning)

```
  # src/explore_persona_space/task_workflow.py, _breadcrumb_fields (line ~1907)
+ _BREADCRUMB_VALUE_TRAILING_PUNCT = ":;,."
  ...
-         fields[key] = value
+         value = value.rstrip(_BREADCRUMB_VALUE_TRAILING_PUNCT)
+         if value and key not in fields:
+             # first NON-EMPTY wins: a later bare `key=` prose substring can
+             # no longer re-bind an already-parsed field to empty/garbage
+             fields[key] = value
```

Planner note: weigh first-non-empty-wins vs last-wins semantics against
existing pinned tests (`tests/test_task_workflow*.py`); trailing-punct strip
must not eat legitimate values (labels are kebab-slugs, paths in `log=` /
`harvest=` never end in `:;,.` by convention — state this in the plan).
Add regression tests reproducing both #1689 shapes (glued colon; prose
`label=` re-bind).

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rn "_breadcrumb_fields" src/ scripts/ .claude/`) and update every
  hit; list them in the plan. (`scripts/tick_triage.py` consumes the shared
  helper — behavior change flows through automatically; verify its pid-probe
  arm against the hardened parser.)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard,
  `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: 2331b34907b3

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py
bug_observed: stage-dispatch breadcrumb label token with glued trailing punctuation (label=wider-lambda-ceilings:) or a prose 'label=' substring re-binding the field to empty makes executing_followup_label fall back to the queue-head unrun label, misidentifying the executing round
why_workflow_gap: _breadcrumb_fields whitespace-splits key=value tokens last-wins with no value hygiene, so prose-glued punctuation or bare key= prose substrings silently misroute the round-label resolution every consumer (watcher on-behalf run-marker, step-4 completion marker, tick_triage pid probe) depends on
proposed_change: harden _breadcrumb_fields token parsing: strip trailing punctuation from values and ignore empty-value rebinds so prose-adjacent tokens cannot misroute executing_followup_label
diff_sketch: |
  + value = value.rstrip(":;,.")
  + if value and key not in fields:
  +     fields[key] = value
  - fields[key] = value
confidence: high
related_task: #1689
<!-- /workflow-fix-candidate -->

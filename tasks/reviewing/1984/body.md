---
title: 'workflow-fix: widen the leading version-stamp stripper to also accept the '
kind: infra
tags:
- wf-fix
- wf-fix-fp:f220fe0a1617
- urgent-main-red
created_at: '2026-08-02T00:03:43Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: src/explore_persona_space/task_workflow.py\n\
  bug_observed: tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers\
  \ is currently red on origin/main because task #1900's epm:same-issue-followup-run\
  \ marker (2026-08-01T03:30:56Z) leads with \"v1 — \" (version stamp + em-dash) and\
  \ parse_followup_note_field's leading-stamp stripper (task_workflow.py ~L3270) accepts\
  \ only the \"v<k>.\" dot+whitespace form, so its explicit \"followup_label: tfmargin-validation-expand\"\
  \ field parses as None.\nwhy_workflow_gap: the field-only parser's stamp-stripper\
  \ is narrower than the note shapes live sessions actually post, so a well-formed\
  \ field-bearing run marker fails the corpus-replay invariant and every intervening\
  \ session's Step 9c gate must re-classify the fleet-wide red.\nproposed_change:\
  \ widen the leading version-stamp stripper to also accept the \"v<k> — \"/\"v<k>\
  \ -\" dash-led form (fields stay explicit — no label inference, preserving the #1111\
  \ field-only decision), and/or corrective re-post of #1900's run marker; add the\
  \ \"v1 — \" shape to the parser tests.\ndiff_sketch: |\n  src/explore_persona_space/task_workflow.py\
  \ (parse_followup_note_field, ~L3270):\n  - strip leading stamp matching r\"^v\\\
  d+\\.\\s+\"\n  + strip leading stamp matching r\"^v\\d+(?:\\.\\s+|\\s*[—–-]\\s+)\"\
  \n  tests/test_workflow_followup_labels.py: add a \"v1 — followup_label: x; ...\"\
  \ parse case;\n  remove/avoid allowlisting (1900, \"2026-08-01T03:30:56Z\") once\
  \ the parser accepts it.\nurgency: main-red\nfailing_test: tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers\n\
  wf_fix: true\nconfidence: high\nrelated_task: #1961\n<!-- /workflow-fix-candidate\
  \ -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1961. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers` red on origin/main: widen the leading version-stamp stripper to also accept the "v<k> — "/"v<k> -" dash-led form (fields stay explicit — no label inference, preserving the #1111 field-only decision), and/or corrective re-post of #1900's run marker; add the "v1 — " shape to the parser tests.

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers is currently red on origin/main because task #1900's epm:same-issue-followup-run marker (2026-08-01T03:30:56Z) leads with "v1 — " (version stamp + em-dash) and parse_followup_note_field's leading-stamp stripper (task_workflow.py ~L3270) accepts only the "v<k>." dot+whitespace form, so its explicit "followup_label: tfmargin-validation-expand" field parses as None.
- **Failing node (router-verified):** `tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers`
- **Confidence (emitter):** high
- verified-at-filing: `uv run pytest tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers -q` -> rc=1 at main @ 9d778420e9 (2026-08-02T00:03:20Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Failing node: `tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- This session carries a `workflow_fix_target:` Provenance line — it MUST
  NOT auto-route its own subagents' workflow-fix candidates (recursion
  guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: f220fe0a1617
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py
bug_observed: tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers is currently red on origin/main because task #1900's epm:same-issue-followup-run marker (2026-08-01T03:30:56Z) leads with "v1 — " (version stamp + em-dash) and parse_followup_note_field's leading-stamp stripper (task_workflow.py ~L3270) accepts only the "v<k>." dot+whitespace form, so its explicit "followup_label: tfmargin-validation-expand" field parses as None.
why_workflow_gap: the field-only parser's stamp-stripper is narrower than the note shapes live sessions actually post, so a well-formed field-bearing run marker fails the corpus-replay invariant and every intervening session's Step 9c gate must re-classify the fleet-wide red.
proposed_change: widen the leading version-stamp stripper to also accept the "v<k> — "/"v<k> -" dash-led form (fields stay explicit — no label inference, preserving the #1111 field-only decision), and/or corrective re-post of #1900's run marker; add the "v1 — " shape to the parser tests.
diff_sketch: |
  src/explore_persona_space/task_workflow.py (parse_followup_note_field, ~L3270):
  - strip leading stamp matching r"^v\d+\.\s+"
  + strip leading stamp matching r"^v\d+(?:\.\s+|\s*[—–-]\s+)"
  tests/test_workflow_followup_labels.py: add a "v1 — followup_label: x; ..." parse case;
  remove/avoid allowlisting (1900, "2026-08-01T03:30:56Z") once the parser accepts it.
urgency: main-red
failing_test: tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers
wf_fix: true
confidence: high
related_task: #1961
<!-- /workflow-fix-candidate -->

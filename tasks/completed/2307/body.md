---
title: 'workflow-fix: Determine which side regressed before touching either. EITHE'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3e2d1ce9d0f6
- urgent-main-red
created_at: '2026-08-14T23:43:24Z'
has_clean_result: false
origin_prompt: '<!-- workflow-fix-candidate v1 -->

  urgency: main-red

  failing_test: tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers

  wf_fix: true

  target_file: scripts/task.py

  bug_observed: Pre-existing trunk red surfaced by the #2295 Step 10d gate''s TG baseline
  leg, and re-surfaced by every intervening session''s gate until routed (the #1701
  -> #1698 shape). The historical-corpus replay test finds 6 run-marker rows whose
  followup_label is unparseable by parse_followup_note_field — tasks 2054 (2026-08-12T19:00:39Z),
  2203 (2026-08-10T15:46:57Z and 2026-08-10T21:57:35Z), 2224 (2026-08-13T03:47:32Z
  and 2026-08-13T03:47:37Z), 2254 (2026-08-14T18:14:08Z).

  why_workflow_gap: A red on main makes every session''s Step 9c / Step 10d gate re-classify
  the same failure, which is fleet-wide per-hour cost; and the two candidate causes
  have opposite fixes, so leaving it unrouted also leaves the diagnosis unmade.

  proposed_change: Determine which side regressed before touching either. EITHER the
  emitting sites wrote malformed labels (fix the emitter AND extend KNOWN_MALFORMED_RUN_MARKERS
  for the six already-landed rows, which cannot be rewritten) OR the parser regressed
  against a legitimate label shape (grep the producing call sites BEFORE tightening
  the regex — the #545 lesson: tightening a parser against live producers converts
  one red into many).

  evidence: /tmp/issue-2295-tg-baseline.txt (identical failure across gate runs 1-3);
  the gated leg was green in both completed runs. Mechanically reproducible: `uv run
  pytest tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers`.

  confidence: high

  related_task: #2295

  <!-- /workflow-fix-candidate -->'
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#2295. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers` red on origin/main: Determine which side regressed before touching either. EITHER the emitting sites wrote malformed labels (fix the emitter AND extend KNOWN_MALFORMED_RUN_MARKERS for the six already-landed rows, which cannot be rewritten) OR the parser regressed against a legitimate label shape (grep the producing call sites BEFORE tightening the regex — the #545 lesson: tightening a parser against live producers converts one red into many).

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** Pre-existing trunk red surfaced by the #2295 Step 10d gate's TG baseline leg, and re-surfaced by every intervening session's gate until routed (the #1701 -> #1698 shape). The historical-corpus replay test finds 6 run-marker rows whose followup_label is unparseable by parse_followup_note_field — tasks 2054 (2026-08-12T19:00:39Z), 2203 (2026-08-10T15:46:57Z and 2026-08-10T21:57:35Z), 2224 (2026-08-13T03:47:32Z and 2026-08-13T03:47:37Z), 2254 (2026-08-14T18:14:08Z).
- **Failing node (router-verified):** `tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers`
- **Confidence (emitter):** high
- verified-at-filing: `uv run pytest tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers -q` -> rc=1 at main @ 6a5900577e (2026-08-14T23:43:11Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `scripts/task.py`
- Failing node: `tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- This session carries a `workflow_fix_target:` Provenance line — it MUST
  NOT auto-route its own subagents' workflow-fix candidates (recursion
  guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: scripts/task.py
- fingerprint: 3e2d1ce9d0f6
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
urgency: main-red
failing_test: tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers
wf_fix: true
target_file: scripts/task.py
bug_observed: Pre-existing trunk red surfaced by the #2295 Step 10d gate's TG baseline leg, and re-surfaced by every intervening session's gate until routed (the #1701 -> #1698 shape). The historical-corpus replay test finds 6 run-marker rows whose followup_label is unparseable by parse_followup_note_field — tasks 2054 (2026-08-12T19:00:39Z), 2203 (2026-08-10T15:46:57Z and 2026-08-10T21:57:35Z), 2224 (2026-08-13T03:47:32Z and 2026-08-13T03:47:37Z), 2254 (2026-08-14T18:14:08Z).
why_workflow_gap: A red on main makes every session's Step 9c / Step 10d gate re-classify the same failure, which is fleet-wide per-hour cost; and the two candidate causes have opposite fixes, so leaving it unrouted also leaves the diagnosis unmade.
proposed_change: Determine which side regressed before touching either. EITHER the emitting sites wrote malformed labels (fix the emitter AND extend KNOWN_MALFORMED_RUN_MARKERS for the six already-landed rows, which cannot be rewritten) OR the parser regressed against a legitimate label shape (grep the producing call sites BEFORE tightening the regex — the #545 lesson: tightening a parser against live producers converts one red into many).
evidence: /tmp/issue-2295-tg-baseline.txt (identical failure across gate runs 1-3); the gated leg was green in both completed runs. Mechanically reproducible: `uv run pytest tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers`.
confidence: high
related_task: #2295
<!-- /workflow-fix-candidate -->

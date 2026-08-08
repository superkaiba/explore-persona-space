---
title: 'daily-fix: fix red main: test_issue1417_render.py::test_render_config_hash_pin
  — reconcile the issue-1417 render-config h'
kind: infra
tags:
- urgent-main-red
created_at: '2026-07-28T16:43:09Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: tests/test_issue1417_render.py\n\
  bug_observed: tests/test_issue1417_render.py::test_render_config_hash_pin fails\
  \ on the main checkout (rc=1, no local dirt on scripts/*1417*): RENDER_CONFIG_HASH_PIN\
  \ no longer matches issue1417_render.render_config_hash(), i.e. the issue-1417 render\
  \ config changed on main (last touch d6bf719ced) without the pin update the test\
  \ mandates via an approved plan amendment.\nwhy_workflow_gap: a live red on main\
  \ forces every intervening session's Step 9c gate to re-classify it as pre-existing;\
  \ the pin/config divergence needs reconciliation (either the plan-amendment pin\
  \ update stranded, or the config change was unapproved).\nproposed_change: reconcile\
  \ the issue-1417 render-config hash pin with the current render_config_hash() under\
  \ the task's must-ask amendment protocol (update the pin with the amendment record,\
  \ or revert the unapproved config change).\ndiff_sketch: |\n  # tests/test_issue1417_render.py\n\
  \  - RENDER_CONFIG_HASH_PIN = \"<stale hash>\"\n  + RENDER_CONFIG_HASH_PIN = \"\
  <current issue1417_render.render_config_hash() value>\"  # amendment: <ref>\nconfidence:\
  \ medium\nrelated_task: #1765\nurgency: main-red\nfailing_test: tests/test_issue1417_render.py::test_render_config_hash_pin\n\
  wf_fix: false\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1765. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_issue1417_render.py::test_render_config_hash_pin` red on origin/main: reconcile the issue-1417 render-config hash pin with the current render_config_hash() under the task's must-ask amendment protocol (update the pin with the amendment record, or revert the unapproved config change).

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** tests/test_issue1417_render.py::test_render_config_hash_pin fails on the main checkout (rc=1, no local dirt on scripts/*1417*): RENDER_CONFIG_HASH_PIN no longer matches issue1417_render.render_config_hash(), i.e. the issue-1417 render config changed on main (last touch d6bf719ced) without the pin update the test mandates via an approved plan amendment.
- **Failing node (router-verified):** `tests/test_issue1417_render.py::test_render_config_hash_pin`
- **Confidence (emitter):** medium
- verified-at-filing: `uv run pytest tests/test_issue1417_render.py::test_render_config_hash_pin -q` -> rc=1 at main @ ea2bda4859 (2026-07-28T16:43:06Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `tests/test_issue1417_render.py`
- Failing node: `tests/test_issue1417_render.py::test_render_config_hash_pin`

## Constraints / invariants

- NON-workflow-surface fix (`wf_fix: false` declared by the parking
  session — the /daily route-2 analogue); scope stays on the named
  target, never `tasks/` state.

## Provenance

- fingerprint: 4759065003df
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: tests/test_issue1417_render.py
bug_observed: tests/test_issue1417_render.py::test_render_config_hash_pin fails on the main checkout (rc=1, no local dirt on scripts/*1417*): RENDER_CONFIG_HASH_PIN no longer matches issue1417_render.render_config_hash(), i.e. the issue-1417 render config changed on main (last touch d6bf719ced) without the pin update the test mandates via an approved plan amendment.
why_workflow_gap: a live red on main forces every intervening session's Step 9c gate to re-classify it as pre-existing; the pin/config divergence needs reconciliation (either the plan-amendment pin update stranded, or the config change was unapproved).
proposed_change: reconcile the issue-1417 render-config hash pin with the current render_config_hash() under the task's must-ask amendment protocol (update the pin with the amendment record, or revert the unapproved config change).
diff_sketch: |
  # tests/test_issue1417_render.py
  - RENDER_CONFIG_HASH_PIN = "<stale hash>"
  + RENDER_CONFIG_HASH_PIN = "<current issue1417_render.render_config_hash() value>"  # amendment: <ref>
confidence: medium
related_task: #1765
urgency: main-red
failing_test: tests/test_issue1417_render.py::test_render_config_hash_pin
wf_fix: false
<!-- /workflow-fix-candidate -->

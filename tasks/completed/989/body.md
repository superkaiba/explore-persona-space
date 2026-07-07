---
title: 'daily-fix: 4xx short-circuit in hub _is_transient_upload_error'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-04T07:12:44Z'
has_clean_result: false
origin_prompt: 'parked — running under workflow_fix_target Provenance / EPM_WORKFLOW_FIX_SESSION
  recursion guard (see .claude/rules/workflow-fix-on-bug.md § Recursion guard). Candidates
  surfaced by the #939 plan grep-audit + implementer prose (NOT auto-routed from this
  session; a future orchestrator pass may file them, dedup grain (target_file, fingerprint)):

  1. src/explore_persona_space/orchestrate/hub.py:1321 _hf_artifact_exists — claimed-urls
  leg full-lists the repo via list_repo_files_complete; same #920 hang class on the
  ~1M-file data repo. Fix: thread the NEW path_in_repo kwarg (landed by #939) per
  cited path. (Also already noted in #939''s epm:clarify.)

  2. scripts/backend_poll.py:1210 _wedged_run_inputs_on_hf — bare list_repo_files(HF_DATA_REPO)
  inside the watcher pod-safety wedge gate (imported by autonomous_session_watch.py).
  Same hang class, runs every 10 min when armed.

  3. scripts/cleanup_pod.py:176 — bare list_repo_files on the model repo (lower severity).

  4. scripts/router_acceptance.py:428 _default_list_hf_repo_files — bare listing;
  injectable seam exists.

  5. scripts/test_integration.py:444 — bare listing on the model repo (integration
  probe).

  6. scripts/build_canonical_persona_pool.py:288 — bare list_repo_files(HF_DATA_REPO)
  post-upload verify (data-prep script, outside the enumerated wf-fix surface).

  7. hub.py _is_transient_upload_error — fuzzy substring scan can bounded-retry a
  4xx whose message embeds ''500''/''502''/''503''/''504'' (deterministic for paths
  containing those digit tr'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

tighten _is_transient_upload_error to short-circuit non-transient on 4xx status codes

## Workflow gap

- **Bug observed:** hub.py _is_transient_upload_error fuzzy substring scan can bounded-retry a 4xx whose message embeds '500'/'502'/'503'/'504' (deterministic for paths containing those digit triplets, e.g. issue504_* buckets)
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

tighten _is_transient_upload_error to short-circuit non-transient on 4xx status codes

## Scope / surfaces

- Primary target: `src/explore_persona_space/orchestrate/hub.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: src/explore_persona_space/orchestrate/hub.py
- fingerprint: f80970bdcc08

parked — running under workflow_fix_target Provenance / EPM_WORKFLOW_FIX_SESSION recursion guard (see .claude/rules/workflow-fix-on-bug.md § Recursion guard). Candidates surfaced by the #939 plan grep-audit + implementer prose (NOT auto-routed from this session; a future orchestrator pass may file them, dedup grain (target_file, fingerprint)):
1. src/explore_persona_space/orchestrate/hub.py:1321 _hf_artifact_exists — claimed-urls leg full-lists the repo via list_repo_files_complete; same #920 hang class on the ~1M-file data repo. Fix: thread the NEW path_in_repo kwarg (landed by #939) per cited path. (Also already noted in #939's epm:clarify.)
2. scripts/backend_poll.py:1210 _wedged_run_inputs_on_hf — bare list_repo_files(HF_DATA_REPO) inside the watcher pod-safety wedge gate (imported by autonomous_session_watch.py). Same hang class, runs every 10 min when armed.
3. scripts/cleanup_pod.py:176 — bare list_repo_files on the model repo (lower severity).
4. scripts/router_acceptance.py:428 _default_list_hf_repo_files — bare listing; injectable seam exists.
5. scripts/test_integration.py:444 — bare listing on the model repo (integration probe).
6. scripts/build_canonical_persona_pool.py:288 — bare list_repo_files(HF_DATA_REPO) post-upload verify (data-prep script, outside the enumerated wf-fix surface).
7. hub.py _is_transient_upload_error — fuzzy substring scan can bounded-retry a 4xx whose message embeds '500'/'502'/'503'/'504' (deterministic for paths containing those digit triplets, e.g. issue504_* buckets); tightening = short-circuit non-transient on 4xx status codes.

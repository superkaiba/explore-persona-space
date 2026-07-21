---
title: 'daily-fix: digest log_tail_excerpt on trigger-dense runs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:265a19743d5d
- daily-auto-filed
created_at: '2026-07-20T06:46:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): raw 5-line log excerpt
  enters orchestrator context on trigger-dense workloads'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-19 parked-candidate sweep (Step C) from a workflow-fix candidate parked on task #1546 (emitting agent: Alternatives critic concern 2; parked under the recursion guard).

## Goal

Make `scripts/poll_pipeline.py` emit a structural digest (pattern counts + exit code + log path) instead of the raw `log_tail_excerpt` field when the issue's workload is trigger-dense.

## Workflow gap

- **Bug observed:** poll_pipeline.py:1118/:1194 deliver a bounded ~5-line RAW log excerpt by construction — the one raw-text channel entering orchestrator context that #1546's digest-only prose discipline cannot remove.
- **Why it is a workflow gap:** on guard/security/refusal-corpus workloads a raw crash-tail excerpt can carry trigger-dense text into the orchestrator turn, the refusal-wedge exposure class #1074/#1098/#1546 progressively closed everywhere else.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n log_tail_excerpt scripts/poll_pipeline.py` → 6 hits (field def :1118, excerpt-construction comment :1194, emission :5389/:5469); `grep -c 'trigger-dense\|trigger_dense' scripts/poll_pipeline.py` → 0 (no trigger-dense branch exists — absence-of-guard claim, in-target 0-hit is the evidence). Landed-fix history: `git log --oneline --since='2026-07-17' -- scripts/poll_pipeline.py` → 6692680d2c (#1546, digest crash-tails on orchestrator poll turns) landed the ORCHESTRATOR-side discipline; the excerpt FIELD itself remains raw (2026-07-19).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up: structural digest — pattern counts + exit code + log path — replacing the raw excerpt for trigger-dense workloads)

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'log_tail_excerpt' scripts/ .claude/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: fc2fcdf39691

Verbatim parked candidate (epm:workflow-fix-candidate on #1546, 2026-07-19T16:14:26Z): "source: prose-followup (Alternatives critic concern 2). target_file: scripts/poll_pipeline.py. proposed_change: emit a structural digest (pattern counts + exit code + log path) instead of the raw log_tail_excerpt field when the issue's workload is trigger-dense (poll_pipeline.py:1118/:1194 deliver a bounded ~5-line raw excerpt by construction — the one raw-text channel entering orchestrator context that #1546's prose discipline cannot remove)."

---
title: 'daily-fix: persist uv PATH for non-login shells (GCE + pod)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c71533e68a74
- daily-auto-filed
created_at: '2026-07-29T07:08:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1739''s recovery upload
  phase died ''uv: command not found'' (exit 127) in a root/setsid shell on GCE; the
  same class recurred pod-side in #1689 (BatchMode ssh probe could not resolve uv)
  — bootstrap_pod.sh exports PATH only transiently and rc-appends only PYTHONPATH;
  the GCE startup script writes no PATH drop-in'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Sources: group-A P6 + group-D P11 (memory `feedback_pod_uv_path.md` documents the pod half as a known gap).

## Goal

Persist `~/.local/bin` on PATH for non-login/ad-hoc shells on both compute lanes so detached/recovery phases resolve uv.

## Workflow gap

- **Bug observed:** #1739 (GCE): a recovery upload phase launched via sudo/setsid died `uv: command not found` exit 127 (~20 min lost to a dead detached phase + a poll cycle); the in-session fix was a transient PATH export. #1689 (pod): a BatchMode ssh probe failed to resolve uv — the recurring class the `feedback_pod_uv_path.md` memory records.
- **Why it is a workflow gap:** bootstrap_pod.sh exports PATH transiently (L154/160/297) and its rc-file append block covers PYTHONPATH only (L363-405); backends/gcp.py's startup script has no /etc/profile.d/ drop-in — every non-login shell re-discovers the gap.
- **Confidence (emitter):** medium-high
- verified-at-filing: `grep -n 'local/bin' scripts/bootstrap_pod.sh` → transient exports at 154/160/297 only; `grep -n 'PYTHONPATH' scripts/bootstrap_pod.sh` → rc-append block at 363-374 (PATH absent); `grep -n 'profile.d' src/explore_persona_space/backends/gcp.py` → 0 hits (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Pod: extend the existing rc-append loop with `export PATH="$HOME/.local/bin:$PATH"`. GCE: one startup-script line writing an /etc/profile.d/eps-uv-path.sh drop-in (note gcp.py already parametrizes inline `( export PATH=... )` wrappers at 1809/2523 — the drop-in removes the need to remember them).

## Scope / surfaces

- Primary targets: `scripts/bootstrap_pod.sh`, `src/explore_persona_space/backends/gcp.py`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: c71533e68a74

- workflow_fix_target: scripts/bootstrap_pod.sh


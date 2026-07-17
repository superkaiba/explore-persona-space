---
title: 'daily-fix: relay pod_lifecycle stderr through runpod.py'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8fd4e88e564d
- daily-auto-filed
created_at: '2026-07-17T06:58:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): runpod.py''s launch leg
  runs subprocess.run(cmd, check=True) (~L831) with no output capture, so a pod_lifecycle.py
  provision failure surfaces as an opaque ''CalledProcessError ... exit status 1''
  with zero diagnostics (#1336 ~02:49Z: a manual provision racing the automated failover
  leg failed with no relayed reason)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1336, ~02:49Z).

## Goal

Make RunPod provision failures diagnosable from the raised exception.

## Workflow gap

- **Bug observed:** runpod.py's launch leg runs subprocess.run(cmd, check=True) (~L831) with no output capture, so a pod_lifecycle.py provision failure surfaces as an opaque 'CalledProcessError ... exit status 1' with zero diagnostics (#1336 ~02:49Z: a manual provision racing the automated failover leg failed with no relayed reason)
- **Why it is a workflow gap:** Fail-fast is project law, but a fail with zero relayed diagnostics forces manual reproduction to learn the reason.
- **Confidence (emitter):** medium
- verified-at-filing: `sed -n '825,835p' src/explore_persona_space/backends/runpod.py` -> subprocess.run(cmd, check=True) at ~L831, no capture/relay (absence claim)

## Proposed change (candidate diff sketch — refine in planning)

capture pod_lifecycle's stderr and re-raise it in the CalledProcessError detail; consider an ownership probe for an in-flight failover-owned provision before a manual one

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/runpod.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 8fd4e88e564d


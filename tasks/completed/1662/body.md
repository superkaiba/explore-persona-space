---
title: 'daily-fix: suffixed follow-up pods terminate on done'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7c016f92f85c
- daily-auto-filed
created_at: '2026-07-24T06:50:28Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): a suffixed follow-up pod
  sat running finished-but-not-terminated behind an ask-gate at 12-13 USD/hr; no completion-side
  teardown contract exists for suffixed pods'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 (transcript sweep). Incident: `pod-1586-b` (a suffixed follow-up pod, ~$12–13/hr) sat RUNNING finished-but-not-terminated behind an ask-gate — its run was complete and artifacts uploaded, but termination waited on a user reply. (The pod is terminated now; the cost was the idle window.)

## Goal

Make "run complete + uploads verified ⇒ terminate the pod" unconditional for suffixed follow-up pods (`pod-<N>-<slug>`), removing the ask-gate: terminating a DONE pod whose artifacts are verified-uploaded is not a user decision anywhere else in the workflow (Step 8 auto-terminates the primary pod on upload-verification PASS).

## Workflow gap

- **Bug observed:** the primary-pod lifecycle auto-terminates at Step 8 after upload-verification PASS, but suffixed follow-up pods launched on inline/override rounds have no equivalent unconditional teardown step — today's session held one idle at ~$12–13/hr awaiting a user reply.
- **Why it is a workflow gap:** the idle-but-billing class (#664 family); the inline-override carve-out prescribes pre-launch signals (`keep-running` tag, `epm:run-launched`) but no completion-side teardown contract for the suffixed pod.
- **Confidence:** medium-high
- verified-at-filing: `pod.py list-ephemeral` at compose time shows only `pod-1586` running (the suffixed pod already terminated — incident window closed, the CLASS remains); the CLAUDE.md inline-override block's pod-safety paragraph covers pre-launch signals + tag removal but contains no "terminate on verified-done, no ask" clause for suffixed pods (read at compose time; absence claim).

## Proposed change (refine in planning)

Add to the inline-override / multi-pod-per-issue contract (CLAUDE.md § Pods + the carve-out block): a suffixed follow-up pod is terminated (`pod.py terminate --issue N --name-suffix <slug> --yes`) immediately once its run completes and uploads verify — no ask-gate; remove the `keep-running` tag in the same step. Optionally a watcher assist: flag a RUNNING suffixed pod whose issue has a newer completion note.

## Scope / surfaces

- Primary target: `CLAUDE.md` (inline-override pod-safety block + § Pods multi-pod contract)

## Constraints / invariants

- Never auto-terminate a pod whose uploads have NOT verified (the durability contract wins); recursion guard applies.

## Provenance

- fingerprint: 7c016f92f85c

- workflow_fix_target: CLAUDE.md

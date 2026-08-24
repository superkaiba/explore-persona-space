---
title: 'Router: auto-chain RunPod rung crashes on 70B intents (gcp.machine_for_intent
  as validator)'
kind: infra
tags: []
created_at: '2026-08-23T19:43:28Z'
has_clean_result: false
origin_prompt: 'Observed during issue #2389 production dispatch 2026-08-23: backend-auto
  + --intent inf-70b crashed the runpod-first rung with ValueError from gcp.machine_for_intent;
  workaround was the explicit --backend runpod pin'
workflow: v1
---
# Router: auto chain's RunPod rung crashes on 70B intents (gcp.machine_for_intent consulted for a RunPod launch)

## Goal

Fix the unified router so a `backend: auto` dispatch with a 70B intent (`inf-70b`, `ft-70b`) launches on the RunPod rung instead of crashing.

## Bug (observed 2026-08-23, issue #2389 production dispatch)

`dispatch_issue.py launch --issue 2389 --intent inf-70b` (backend absent -> auto) crashed with `ValueError: no GCP machine-type for intent 'inf-70b'` and LAUNCHER_RC=4. Trace: `router.route` -> `_auto_route` (router.py:5738) -> `_attempt_runpod_lane` (router.py:5563) -> `_runpod_terminal_rung` (router.py:4058) -> `gcp.machine_for_intent` (gcp.py:726) raise.

The RUNPOD rung calls the GCP machine-mapping helper as an intent validator. RunPod itself fully supports the intent (`inf-70b` = 8x H100 per the pod intent table; the pods.md doc says "RunPod covers H200 / 70B paths"), so the auto chain is structurally unable to dispatch the exact workloads only RunPod can serve. The documented residual gap in `.claude/skills/issue/steps/10-step-6.md` ("(a) 70B intents ... need the explicit `--backend runpod` override") covers the GCP lane's inability, not an auto-chain crash on its runpod-FIRST rung (#2054 made runpod the head of the chain; the coupling predates that).

## Fix sketch

In `_runpod_terminal_rung` (and any other RunPod-lane call site of `gcp.machine_for_intent`), stop consulting the GCP table for intent validation — validate against the RunPod intent table (`pod_lifecycle` / `runpod_api` intent specs) or simply thread the intent through to `pod.py provision`, which already resolves it. A GCP-unmappable intent must not fail the RUNPOD rung; it should at most skip a GCP rung (already absent under #2028).

## Repro

`uv run python scripts/dispatch_issue.py launch --issue <any> --intent inf-70b --repo-branch <branch> --workload-cmd 'echo hi'` with backend-absent frontmatter -> ValueError before any provisioning. Workaround in the field: `--backend runpod` explicit pin (used for the #2389 production dispatch, marker-recorded there).

## Acceptance

- An auto-route dispatch with `--intent inf-70b` reaches `RunPodBackend.launch` (or a capacity miss falls through cleanly) instead of raising `ValueError`.
- A regression test pinning the auto chain x 70B-intent path (mirror of `tests/test_router.py::test_runpod_first_capacity_miss_falls_through_then_terminal_retry`).
- The Step 6b residual-gap prose updated to drop the 70B row from the must-override list once the fix lands (or scoped to the explicit-gcp-pin case only).

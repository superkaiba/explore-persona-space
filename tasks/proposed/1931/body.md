---
title: 'daily-fix: pod bootstrap retry + fail-loud provision verdict'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7e3bdaeecfe2
- daily-auto-filed
created_at: '2026-07-31T07:00:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): pod-1773-regsteer first
  bootstrap died exit 100 leaving /workspace empty with no retry and no provision-level
  failure signal; the gap surfaced only when the pilot launch failed PILOT_MISSING.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 (problem sweep; miner-7 P10, session 0ac15c23 / pod-1773-regsteer).

## Goal

Make a failed pod bootstrap loud at provision time: one automatic bootstrap retry on nonzero exit, and a fail-loud provision verdict when bootstrap still exits nonzero — instead of returning success-shaped provision output over an empty /workspace.

## Workflow gap

- **Bug observed:** pod-1773-regsteer's first bootstrap died exit 100 leaving `/workspace/explore-persona-space` absent, with no retry and no provision-level failure signal; the gap surfaced only when the pilot launch failed `cd: /workspace/explore-persona-space: No such file or directory` (PILOT_MISSING). A manual bootstrap re-run succeeded immediately (transient apt/network-class failure).
- **Why it is a workflow gap:** provision's contract is a ready pod; a silently-failed bootstrap converts every downstream launch into a confusing failure that costs a diagnosis round before anyone re-runs bootstrap.
- **Confidence (emitter):** medium (exit code read from the provision log excerpt in-session; unverified hypothesis — verify at plan time: whether `pod_lifecycle._bootstrap`'s rc is checked/propagated by the provision caller)
- verified-at-filing: `grep -c 'retry' scripts/bootstrap_pod.sh` → 0 (no retry in the script); `grep -n 'def _bootstrap' scripts/pod_lifecycle.py` → :744 (the caller-side rc handling is the plan-time read).

## Proposed change (candidate diff sketch — refine in planning)

In `scripts/pod_lifecycle.py` (provision path): on `_bootstrap` rc != 0, retry ONCE; if still nonzero, print a loud BOOTSTRAP-FAILED verdict and exit nonzero (or mark the provision result failed) rather than returning success-shaped output. Keep `--no-bootstrap` semantics unchanged.

## Scope / surfaces

- Primary target: `scripts/pod_lifecycle.py` (+ `scripts/bootstrap_pod.sh` only if the retry belongs script-side)

## Constraints / invariants

- Never mask a persistent bootstrap failure (retry once, then fail loud — no infinite retry).
- Existing pod tests stay green.

## Provenance

- fingerprint: 7e3bdaeecfe2

- workflow_fix_target: scripts/pod_lifecycle.py, scripts/bootstrap_pod.sh
- origin: /daily 2026-07-30 miner-7 P10 (session 0ac15c23, pod-1773-regsteer)

---
title: 'daily-fix: pod.py sync env GITHUB_TOKEN refresh leg'
kind: infra
tags:
- wf-fix
- wf-fix-fp:511e3c3bbba2
- daily-auto-filed
created_at: '2026-07-16T07:21:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): pod-1333 git fetch 403''d
  on a stale GITHUB_TOKEN, sync env didn''t fix it, attempts 5-7 ran on manual git-bundle
  sideload; #1315 hit the same shape; #1361 shipped doc only'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

`pod.py sync env` (or a `keys --refresh-token` leg) re-pushes GITHUB_TOKEN from the VM .env to the pod and verifies with a `git ls-remote` probe, so relaunches stop depending on manual bundle-sync.

## Workflow gap

- **Bug observed:** pod-1333's git fetch 403'd on a stale GITHUB_TOKEN, `pod.py sync env` also failed to fix it, and attempts 5-7 ran on a manual git-bundle sideload (fc2b61b7 16:50/17:02/18:38Z); #1315's pod hit the same 403-with-valid-token shape (c7b67f30 19:16Z). #1361 shipped the gotchas DOC entry only — no code path re-pushes a fresh token.
- **Why it is a workflow gap:** token push happens once at bootstrap (bootstrap_pod.sh step 3), and no pod.py subcommand refreshes or verifies the pod-side GITHUB_TOKEN afterwards — so a rotated/stale token strands every subsequent pod git operation until a human sideloads.
- **Severity:** medium
- verified-at-filing: `grep -n 'GITHUB_TOKEN' scripts/pod.py scripts/pod_lifecycle.py scripts/pod_config.py` → 0 hits (no pod.py token handling — absence confirmed); `grep -n 'GITHUB_TOKEN' scripts/bootstrap_pod.sh` → L47/L61-63 (credential helper reads it from pod .env), L197-210 (step 3 pushes .env once at bootstrap) — bootstrap-only presence confirmed; `pod.py` sync entry `cmd_sync` at scripts/pod.py:163, `env` branch at L174 (2026-07-16 UTC).

## Proposed change (refine in planning)

Add a token-refresh leg to `scripts/pod.py`: either fold into `sync env` (cmd_sync L163, env branch L174) or add `keys --refresh-token` — re-push the VM `.env`'s GITHUB_TOKEN to the pod's `$REMOTE_DIR/.env` (reusing bootstrap_pod.sh's step-3 shape) and verify with an on-pod `git ls-remote origin` probe, failing loud with the 403 body when the token is genuinely invalid. Document the ladder step in the #1361 gotchas entry so the doc points at the code path instead of manual bundle-sync.

## Scope / surfaces

- Primary target: `scripts/pod.py` (cmd_sync L163 / keys path)
- Secondary: `scripts/bootstrap_pod.sh` (reuse the step-3 push shape, L197-210); `.claude/rules/gotchas.md` (#1361 entry cross-link)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- No token value ever printed/logged (secrets hygiene; grep-for-secrets before commit).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 511e3c3bbba2

- workflow_fix_target: scripts/pod.py

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: fc2b61b7 (#1333) 16:50/17:02/18:38Z (batch 05 P8); c7b67f30 (#1315) 19:16Z (batch 09 P9).

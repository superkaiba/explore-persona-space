---
title: 'daily-fix: IAP tunneling in GcpBackend.fetch_results'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ed354970d76a
- daily-auto-filed
created_at: '2026-07-17T06:56:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): GcpBackend.fetch_results''
  best-effort ssh/scp path fails rc=255 with ''External IP address was not found;
  defaulting to using IAP tunneling'' on external-IP-less GCP VMs — the fallback results
  fetch is silently useless on such VMs (benign when HF/git carry the authoritative
  copy, but it is the crash-forensics path)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1005 session, ~03:48Z).

## Goal

Make the GCP best-effort results fetch work on IAP-only (external-IP-less) VMs.

## Workflow gap

- **Bug observed:** GcpBackend.fetch_results' best-effort ssh/scp path fails rc=255 with 'External IP address was not found; defaulting to using IAP tunneling' on external-IP-less GCP VMs — the fallback results fetch is silently useless on such VMs (benign when HF/git carry the authoritative copy, but it is the crash-forensics path)
- **Why it is a workflow gap:** fetch_results is a crash-forensics safety net; a silent rc=255 on a common VM shape removes it exactly when needed.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'def fetch_results' src/explore_persona_space/backends/gcp.py` -> L6184; `grep -n tunnel src/explore_persona_space/backends/gcp.py` -> 0 hits (absence claim); incident text in #1005 transcript (rc=255 + IAP defaulting notice)

## Proposed change (candidate diff sketch — refine in planning)

thread --tunnel-through-iap into the fetch_results gcloud ssh/scp invocations (or detect the no-external-IP case and add the flag)

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: ed354970d76a


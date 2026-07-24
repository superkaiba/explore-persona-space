---
title: 'daily-fix: provision preflight checks account SSH keys'
kind: infra
tags:
- wf-fix
- wf-fix-fp:32fb78b2e709
- daily-auto-filed
created_at: '2026-07-24T06:48:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): fresh pods refused SSH
  publickey because the VM key was dropped from the shared team account key list by
  fellows onboarding and provision has no account key-list check'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 (transcript sweep). Incident: every FRESH RunPod pod provisioned during a window on 2026-07-23 refused SSH with `Permission denied (publickey)` (5 tool_result firing events in one session; also hit in the #1586 session) — root cause was the VM's `id_ed25519` public key having been dropped from the TEAM account's key list by the fellows-cluster onboarding. Diagnosis burned pod-billing minutes per pod because the failure only surfaces after provision+bootstrap.

## Goal

Verify at provision time that the VM's SSH public key is present in the RunPod team account's key list, failing loud BEFORE creating the pod.

## Workflow gap

- **Bug observed:** `pod.py provision` assumes the account key list contains the VM key; a foreign mutation of the shared team account (fellows onboarding) silently breaks SSH on every subsequent fresh pod, and the failure mode (publickey denied post-provision) doesn't name the cause.
- **Why it is a workflow gap:** the account is shared with the fellows fleet, so foreign key-list mutations are now a live hazard class; `pod.py keys --verify` verifies keys ON pods, not the account-level list a fresh pod is seeded from.
- **Confidence:** medium-high
- verified-at-filing: `grep -n "keys\|ssh_key\|pubkey" scripts/pod_lifecycle.py` → provision path references `~/.ssh/id_ed25519` (line 829) with no account-key-list check (absence claim, in-target 0-hit for an account-list verification) (2026-07-24 UTC).

## Proposed change (refine in planning)

Provision preflight: fetch the account's public-key list via the RunPod API, assert the VM's `id_ed25519.pub` is present (re-add or fail loud with the remediation named), before `create_pod`.

## Scope / surfaces

- Primary target: `scripts/pod_lifecycle.py` (+ `scripts/runpod_api.py` if a key-list query is missing)

## Constraints / invariants

- Fail loud, never auto-mutate the shared team account's key list without logging it.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 32fb78b2e709

- workflow_fix_target: scripts/pod_lifecycle.py

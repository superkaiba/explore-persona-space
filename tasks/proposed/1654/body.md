---
title: 'daily-fix: HF upload-quota preflight at real scale'
kind: infra
tags:
- wf-fix
- wf-fix-fp:aa6b52823185
- daily-auto-filed
created_at: '2026-07-24T06:48:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): small probe uploads pass
  while production-scale checkpoint uploads 403 on billing state so the failure surfaces
  after training completes'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 (transcript sweep). Incident on #1586: HF returned 403 "setup automatic credit recharge" on PRODUCTION-scale checkpoint uploads while small (~2MB) probe uploads PASSED — the run's upload phase died at scale after the small-probe preflight read green, and the user had to resolve billing state by hand mid-run ("why is it bugging", "so is it done being handled?").

## Goal

Make the upload-side preflight catch quota/billing-state failures that only manifest at production scale: probe at a representative size and/or read the account's billing/quota state directly, instead of trusting a KB–MB canary.

## Workflow gap

- **Bug observed:** the preflight's HF check verifies reachability (and disk quotas locally) but has no upload-quota/billing probe at representative size — a 2MB probe passes while a multi-GB LFS/Xet upload 403s on billing state, so the failure surfaces only after training completes, at the most expensive moment.
- **Why it is a workflow gap:** upload-before-delete is the project's durability contract; a false-green preflight converts a billing hiccup into a checkpoint-stranding incident.
- **Confidence:** medium (the right probe size/mechanism needs design — a whoami/billing API read may beat a large probe upload).
- verified-at-filing: `grep -n "probe|upload|quota|403" src/explore_persona_space/orchestrate/preflight.py` → probes are disk-writable-bytes canaries only (MooseFS quota class); no HF upload-quota/billing probe exists (absence claim, in-target 0-hit) (2026-07-24 UTC).

## Proposed change (refine in planning)

Add an upload-quota leg to preflight (or the upload-verifier's pre-upload path): either query the HF billing/quota state via API, or a representative-size (~100MB, deleted after) probe to the target repo class, failing loud with the recharge remediation named.

## Scope / surfaces

- Primary target: `src/explore_persona_space/orchestrate/preflight.py` (+ `.claude/rules/upload-policy.md` note)

## Constraints / invariants

- Fail fast, never fallback-silently; the probe must not spam repos with garbage blobs (clean up after itself).
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: aa6b52823185

- workflow_fix_target: src/explore_persona_space/orchestrate/preflight.py

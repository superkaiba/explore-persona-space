---
title: 'daily-fix: detached-VM probe template brackets; choom=failed'
kind: infra
tags:
- wf-fix
- wf-fix-fp:15518cc27242
- daily-auto-filed
created_at: '2026-07-30T07:08:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): #1482''s VM re-sweep was
  earlyoom-killed at 32 GiB RSS after choom=failed, and the kill hid ~50 min behind
  a false-ALIVE unbracketed pgrep self-match — the SKILL''s detached-phase probe snippet
  does not restate the bracket rule, and choom=failed on a big-RSS phase proceeds
  unprotected'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner D-P2 (session ff4119b7, #1482; partially probed)).

## Goal

A detached VM phase's liveness must be readable (bracketed probes) and its earlyoom protection must not silently stay off for big-RSS phases.

## Workflow gap

- **Bug observed:** earlyoom SIGTERMed the sweep at VmRSS 32,254 MiB (choom had failed at launch, recorded and proceeded per current rule); two liveness probes read ALIVE because the unbracketed pgrep matched its own wrapper; detection took ~50 min, then the leg was re-dispatched to GCP.
- **Why it is a workflow gap:** the choom=failed proceed-unprotected default is fine for small phases but not >= 16 GiB RSS ones, and the ad-hoc probe template does not restate the gotchas bracket rule where probes get composed.
- **Confidence (emitter):** medium
- verified-at-filing: miner probe: `grep -n choom .claude/skills/issue/SKILL.md` -> L6505-6590 (best-effort by design; no bracket mandate in the probe snippet; no big-RSS retry) (2026-07-30).

## Proposed change (refine in planning)

Amend § Detached VM-side long compute phases per the two-part change above; cross-ref the gotchas ownership-probe entry (extended tonight, route 1).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 15518cc27242

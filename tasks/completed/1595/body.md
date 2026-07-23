---
title: 'daily-fix: persist resume state before multi-hour pod stop'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e6462a51211f
- daily-auto-filed
created_at: '2026-07-22T06:46:22Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): a stopped RunPod volume
  vanished provider-side despite keep-running and the 7-day idle window, losing resume
  state'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-21 from the #1112 vanished-stopped-pod incident (transcript 24ae2158).

## Goal

Treat a STOPPED RunPod volume as NON-durable for multi-hour parks: require uploading the run's done-JSONs / resume sentinels to HF before `pod.py stop` whenever a park may outlast ~1 hour.

## Workflow gap

- **Bug observed:** pod-1112 was stopped 2026-07-21T07:25Z with volume preserved, `keep-running` tag set, and "Will auto-terminate after 7 days idle" — yet ~22h later both the ephemeral record and the live RunPod API showed it GONE (`{"data": {"pod": null}}`), losing the volume (done-JSONs, resume state) and forcing a full re-run. `keep-running` was set and the 7-day window hadn't elapsed, so project janitors shouldn't have fired; most consistent with RunPod-side reclaim of a stopped pod.
- **Why it is a workflow gap:** the pause/stop recipes treat a stopped pod's volume as durable park state; nothing requires persisting resume sentinels off-pod before a multi-hour stop. Persist-by-default covers artifacts but not RESUME STATE explicitly.
- **Confidence:** medium (root cause of the vanish is not transcript-derivable; the mitigation is valid regardless — the science loss was bounded only because smoke/capture artifacts happened to be on HF).
- verified-at-filing: `grep -n 'stop\|durab\|resume state\|volume' .claude/rules/pod-config.md` → no stopped-volume durability guidance (2 unrelated hits: a stale-port polling incident, a registry migration note — absence claim, in-target probe 2026-07-22). `git log --oneline --since='7 days ago' -- .claude/rules/pod-config.md` → no such duty landed.

## Proposed change (candidate diff sketch — refine in planning)

Add to `.claude/rules/pod-config.md` (and/or the `/issue` pause recipe § User pause affordance): before `pod.py stop` for a park that may outlast ~1h, upload the run's done-JSONs + resume sentinels to the HF data repo (extends persist-by-default to resume state); on resume, prefer the off-pod copies. A stopped RunPod volume can be reclaimed provider-side despite `keep-running` + the 7-day idle window.

## Scope / surfaces

- Primary target: `.claude/rules/pod-config.md`
- Possible sibling: `.claude/skills/issue/SKILL.md` § User pause affordance (the planner decides whether to touch it).

## Constraints / invariants

- Workflow-surface only; consistent with `.claude/rules/upload-policy.md` persist-by-default.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: e6462a51211f

- workflow_fix_target: .claude/rules/pod-config.md

Origin evidence: transcript 24ae2158 — 2026-07-21T07:25Z ("Stopped. Will auto-terminate after 7 days idle") vs 2026-07-22T05:36Z ("No ephemeral pod recorded for issue 1112" + live-API `{"data": {"pod": null}}`).

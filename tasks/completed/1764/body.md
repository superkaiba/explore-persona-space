---
title: 'daily-fix: sudo tail for root-owned GCE workload log probes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:df9fae0a86f5
- daily-auto-filed
created_at: '2026-07-28T07:05:14Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): a detached-phase diagnostic
  on #1738''s GCE instance hit ''tail: cannot open /workspace/logs/issue-1738.log:
  Permission denied'' — the workload log is root-owned on GCE and the probe recipe
  does not prescribe sudo, leaving the liveness diagnostic partially blind'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session 77ee3ced (#1738 monitoring), 2026-07-28T06:04Z (miner G P9b).

## Goal

Detached-phase log probes on GCE must be able to read root-owned workload logs.

## Workflow gap

- **Bug observed:** the phase-0 manifest build ran ~99% CPU with the log mtime frozen ~50 min; the diagnostic's `tail` on the workload log failed Permission denied (non-root probe against a root-owned file), so the liveness read fell back to pid/CPU only.
- **Why it is a workflow gap:** the detached-phase probe recipe never mentions sudo for GCE root-owned logs (compose-time grep: all 9 `sudo -n` hits in SKILL.md are choom calls; zero are log reads).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'sudo tail\|sudo -n' .claude/skills/issue/SKILL.md` -> 9 hits, all `choom` (compose time 2026-07-28).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/skills/issue/SKILL.md` (detached VM-side long-compute probe recipe / GCE monitoring block): prescribe `sudo -n tail -50 <log>` (fallback `sudo -n cat`) when a workload-log read fails EACCES on a GCE instance; note the log ownership difference vs RunPod pods.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: df9fae0a86f5

- workflow_fix_target: .claude/skills/issue/SKILL.md

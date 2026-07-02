---
title: 'workflow-fix: watcher CPU/memory-pressure escalation arm for the shared VM'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a0a8c9d4fc3a
created_at: '2026-07-02T07:20:35Z'
has_clean_result: false
origin_prompt: 'PM-chat VM triage 2026-07-02: hours at load 186-226 + earlyoom SIGTERM
  sweeps, invisible to all watchers. Fix: escalate-only PSI/load pass in autonomous_session_watch.py
  with per-issue attribution.'
---
## Overview / Motivation

Auto-filed from the 2026-07-02 shared-VM overload incident (load 186-226 on 32 cores for hours; earlyoom SIGTERM sweeps at <10% mem-avail silently killed 4-7GB analysis workers; a checkpointed sweep's 60s layers stretched to 64min/2.2h). The fleet has disk guards and pod janitors but NO CPU/memory-pressure observability arm.

## Goal

Add a CPU/memory-pressure pass to the autonomous-session watcher: when load or PSI crosses a threshold, write an attribution row (top jobs -> owning issue/session) to a sidecar JSONL + a deduped Telegram escalation, mirroring the disk-guard escalate-only design. Warn-only; never auto-kill.

## Workflow gap

- **Bug observed:** hours at 6x CPU oversubscription + earlyoom kill sweeps were invisible until a user asked why a job was slow; silent SIGTERM deaths (exit 143, no traceback) were misattributed for hours.
- **Why it is a workflow gap:** the watcher watches sessions/pods/disk but not the shared box's CPU/memory pressure, so fleet-wide degradation has no detection or attribution channel.
- **Confidence (emitter):** medium

## Proposed change (refine in planning)

- `scripts/autonomous_session_watch.py`: new escalate-only pass reading /proc/loadavg + /proc/pressure/{cpu,memory}; threshold (e.g. load > 1.5x nproc for 2 consecutive ticks) -> sidecar row (.claude/cache/cpu-guard-events.jsonl) attributing top-CPU processes to issues/sessions + deduped Telegram push. Also surface earlyoom kills (journal grep) so silent SIGTERMs are attributed.
- Document in `.claude/rules/background-automation.md`.

## Scope / surfaces

- Primary: `scripts/autonomous_session_watch.py`, `.claude/rules/background-automation.md`.

## Constraints / invariants

- Escalate-only (no kills, no renice). Fail-soft on missing PSI. Deduped like the disk-guard bands.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- origin: PM-chat VM triage 2026-07-02 (/tmp/vm_load_diagnosis.md)

---
title: 'workflow-fix: per-pod pod-safety predicate for multi-round issues'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2cec7073efd3
created_at: '2026-08-01T06:36:27Z'
has_clean_result: false
origin_prompt: 'Surfaced prose follow-up from inline subagent lasttoken-repool-1768
  on #1768: watcher pod-safety auto-stopped pod-1768-lt mid-fits on a multi-round
  issue (sibling status-changed postdated run-launched); fix = per-pod predicate refinement
  + tag-by-default guidance for concurrent inline rounds'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced on task #1768 (emitting agent: experiment-implementer `lasttoken-repool-1768`, completion report 2026-08-01; incident ~05:05Z).

## Goal

Make the watcher pod-safety pass's inferred follow-up predicate safe on MULTI-ROUND issues: a sibling round's routine status transition must not flip protection off another round's healthy live pod.

## Workflow gap

- **Bug observed:** watcher pod-safety pass auto-stopped pod-1768-lt mid-fits because a sibling round epm:status-changed postdated the round epm:run-launched flipping the inferred predicate off on a multi-round issue
- **Why it is a workflow gap:** the predicate compares the newest follow-up signal (`epm:run-launched`/`epm:followup-scope`/`epm:free-analysis-followup-run`) against the newest done-transition (`epm:promoted`/`epm:status-changed`) at ISSUE grain; on an issue running concurrent/serial follow-up rounds (three inline pods + a session loop on #1768 tonight), routine round transitions post `epm:status-changed` continually, so any pod whose `run-launched` predates the latest sibling transition loses inferred protection — only the timestamp-independent issue-wide `keep-running` tag shields it, and the standing inline-round guidance (this orchestrator's own briefs included) says NO tag to avoid blocking sibling teardowns (#1485). The two protections are mutually exclusive at current grain.
- **Consequence realized:** pod-1768-lt auto-STOPPED at ~05:05Z with 142/216 fit cells done (timeline as reported by the emitting agent — `unverified hypothesis — verify at plan time:` exact marker timestamps 04:11:11Z / 22:34Z; the MECHANISM is confirmed against the predicate's own docstring). Recovered by tag+resume with no data loss, ~1h delay.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "run-launched\|followup-scope\|free-analysis-followup-run\|keep-running" scripts/autonomous_session_watch.py` → 8 hits (2026-08-01); per-target: `scripts/autonomous_session_watch.py` — L202-209 document exactly the newest-signal-vs-done-transition predicate (`_SESSION_FOLLOWUP_SIGNAL_KINDS`) and the "no keep-running tag (the explicit user override)" arm; context read: the predicate is issue-grain by construction, no per-pod matching exists (gap present, not landed-fixed). `CLAUDE.md` carries the inline-round pod-safety clause that prescribes marker-or-tag (the guidance half of the fix).

## Proposed change (candidate diff sketch — refine in planning)

diff_sketch: |
  scripts/autonomous_session_watch.py (pod-safety pass):
  - predicate: newest follow-up signal on the ISSUE vs newest done-transition on the ISSUE
  + per-POD refinement: an `epm:run-launched` whose note names the pod (name/suffix — the
  +   notes already do, e.g. "pod-1768-lt ... provisioned") shields THAT pod while the pod is
  +   RUNNING and younger than a staleness ceiling, regardless of later sibling
  +   status-transitions; issue-grain inference retained as fallback for unnamed launches
  CLAUDE.md (user-chat inline carve-out, pod-safety pre-launch signals) + .claude/skills/issue/SKILL.md 9a-ter:
  + document the multi-round exposure; until the per-pod predicate lands, concurrent inline
  +   rounds SET keep-running at provision + remove at completion, with the surgical
  +   --name-suffix teardown noted as the sibling-safe path under the issue-wide tag

## Scope / surfaces

- Primary targets: `scripts/autonomous_session_watch.py`, `CLAUDE.md`
- Also check `.claude/skills/issue/SKILL.md` § 9a-ter pod-safety block and `scripts/pod_lifecycle.py` (#1485 tag semantics) for consistency; grep `grep -rn "keep-running" scripts/ .claude/ CLAUDE.md` and reconcile every guidance site.

## Constraints / invariants

- Workflow-surface only. The watcher's escalate-only and bounded-stop contracts unchanged; fail toward KEEP (a mis-shielded pod bills, a mis-stopped pod loses work — the wedged-owner escalation arm #1582 remains the billing backstop).
- `scripts/workflow_lint.py --check-asks` passes; tests pinning watcher invariants updated alongside.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py,CLAUDE.md
- fingerprint: 2cec7073efd3

Verbatim surfaced prose (agent completion report, 2026-08-01):

> INCIDENT you should know about: pod-1768-lt was auto-STOPPED mid-fits at ~05:05Z, 142/216 cells done. Cause: a SIBLING round's epm:status-changed at 04:11:11Z is newer than my epm:run-launched (22:34Z), which flips the watcher pod-safety pass's inferred follow-up predicate off — only the timestamp-independent keep-running tag shields that window, and the brief specified NO tag. Recovered with no data loss: tagged, resumed, harvested 46 complete arms, relaunched (resume predicate skipped all 46), remaining 26 finished rc=0. Worth considering whether inline rounds on a multi-round issue should tag by default.

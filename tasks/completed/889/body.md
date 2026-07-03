---
title: 'workflow-fix: pre-dispatch triage gate for external audit markers'
kind: infra
tags:
- wf-fix
- wf-fix-fp:62c18f2ffc26
created_at: '2026-07-02T21:00:19Z'
has_clean_result: false
origin_prompt: 'PM-chat root-cause 2026-07-02: #779 launched the unfixed grid 4 min
  after phase end without triaging 10 queued external audit markers.'
---
## Overview / Motivation
Auto-filed from the 2026-07-02 #779 unfixed-grid launch: a PM-chat audit posted 7 vectorization fixes + additions as task-log markers over 5h; the session's orchestrator dispatched the CPU grid 4 minutes after finishing its prior phase without triaging them, launching an 18-20h serial grid (killed by PM-chat at 12 min; measured-benchmark audit at /tmp/issue779_grid_audit.md, marker 16:12Z; directive marker v79 20:55Z).

## Goal
Add a pre-dispatch check to the /issue stage-dispatch path: before launching any compute stage (pod, GCP, or VM-side phase), scan the task's events.jsonl for UNADDRESSED external audit/directive/advisory markers posted since the last stage-dispatch, and triage them (apply / explicitly defer with reason in the launch marker) BEFORE dispatching.

## Workflow gap
- **Bug observed:** external audit markers (16:12Z-17:47Z, 10 markers incl. a measured 18-20h-vs-1.5h wall-time finding + required fixes) were unread at the 20:46Z grid dispatch; the launch note claimed "vectorized" while the fixes were unapplied.
- **Why it is a workflow gap:** cross-session markers are the sanctioned advisory channel, but no SKILL step requires reading them at the one moment they matter (pre-dispatch); the channel is a mailbox with no read-gate.
- **Confidence (emitter):** high

## Proposed change (refine in planning)
.claude/skills/issue/SKILL.md stage-dispatch/step-6 (and the followup-loop equivalents): before dispatching a compute stage, list events.jsonl markers newer than the previous stage-dispatch; for each external (non-self-authored) directive/audit marker, either apply it or record an explicit deferral reason in the launch marker. A one-line "external-markers triaged: N applied / M deferred (reasons)" field in the stage-dispatch note makes it auditable.

## Scope / surfaces
- Primary: `.claude/skills/issue/SKILL.md`; possibly `.claude/skills/issue-tick/SKILL.md` note.

## Constraints / invariants
- workflow_lint --check-asks passes; no new AskUserQuestion gates (auto-continue preserved); triage is bounded (markers since last dispatch only).

## Provenance
- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: PM-chat root-cause 2026-07-02 (#779 unfixed-grid launch)

---
title: 'daily-fix: escalate keep-running pods with wedged owners'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b7f498313e7f
- daily-auto-filed
created_at: '2026-07-21T06:43:14Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-20 problem sweep (route 2): a RUNNING pod shielded
  by the keep-running tag billed idle ~2.5 days while its owning Happy wrapper was
  wedged (0% CPU, 3.7 days) holding the #1345 registration and worktree; no watcher
  pass covers RUNNING+keep-running+wedged-owner and recovery was fully manual after
  Thomas asked whether the pod was still running'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-20 from the transcript problem sweep (interactive session f8aebc42, 2026-07-21 00:48–03:08Z): task #1345's `onpolicy-assistant-story` inline-override round computed on 07-18 was stranded un-folded for ~2.5 days because its owning Happy wrapper wedged, while its `keep-running`-tagged pod idled at 0% GPU and billed the whole time. Thomas had to notice and ask "are you sure it's not running still?".

## Goal

Close the watcher blind spot where a wedged (0%-CPU, multi-day) session wrapper keeps holding an issue registration + worktree while a `keep-running`-tagged RUNNING pod bills with no live work — detect and escalate (or recover) within hours, not days.

## Workflow gap

- **Bug observed:** the #1345 owning wrapper (pid 2200056) was "0% CPU, up 3.7 days, sleeping — still holding the ~#1345 registration and worktree but doing nothing"; its pod had been silent since 2026-07-18 07:27Z (~2.5 days, 0% GPU, billing). No watcher pass surfaced either: the pod-safety pass is fully shielded by the `keep-running` tag (removed only at recovery, commit `c4a6c4619` on 07-21); the zombie-wrapper pass keys on "no inner Claude process ≥2h" (the wedged wrapper still had one); the stale-registration pass keys on ≥12h transcript idle but evidently did not unhold the wedge or the pod. Recovery was fully manual (stop wrapper → clear registration → respawn → harvest → terminate pod), prompted by Thomas.
- **Why it is a workflow gap:** `keep-running` + dead owner is currently invisible by construction — the documented residue ("a crashed run leaves the tag and the pod bills until manual removal") relies on `pod.py audit-stale`, which only auto-terminates EXITED pods, never a RUNNING idle one. A tag whose owning session is provably wedged should at minimum ESCALATE (Telegram + sidecar) with the idle evidence.
- **Confidence (emitter):** medium-high (incident fully reconstructed; exact pass predicates to blame need the code read)
- verified-at-filing: incident is transcript-evidenced (session f8aebc42; #1345 events: `epm:run-launched` 07-18, `remove-tag keep-running` + `epm:pod-terminated` 07-21 — `git log --oneline` today shows `c4a6c4619 task #1345: remove-tag keep-running` and `980f9eb09c task #1345: epm:pod-terminated`). This is an absence-of-coverage claim about pass predicates in `scripts/autonomous_session_watch.py` — the enumerated passes (pod-safety keep-running shield, zombie-wrapper inner-process predicate, stale-registration idle unregister) are documented in CLAUDE.md § background automation; no pass covers RUNNING+keep-running+wedged-owner (n/a for a single grep — the claim is a cross-pass coverage hole, to be confirmed by the spawned session's planner against the pass code) (2026-07-21).

## Proposed change (candidate diff sketch — refine in planning)

Add a watcher pass (or extend pod-safety): for each RUNNING pod whose owning task carries `keep-running`, when the owning session's wrapper/transcript is provably wedged (0% CPU for >N h, or transcript idle >N h) OR no live session maps to the issue, ESCALATE (Telegram + sidecar row naming the pod, tag, idle evidence, and the recovery recipe) — escalate-only first; any auto-stop arm needs its own guard rails.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (+ `.claude/rules/background-automation.md` doc sync)

## Constraints / invariants

- Never auto-stop on uncertain evidence; escalate-only default (the #770 wedge-arm precedent: auto-terminate only the provably-safe matured case).
- The `keep-running` tag stays an explicit override for LIVE work — the new pass keys on owner-deadness, not on the tag alone.

## Provenance

- fingerprint: b7f498313e7f

- workflow_fix_target: scripts/autonomous_session_watch.py

Origin evidence (transcript-mined, session f8aebc42, 2026-07-21 ~01:15Z): "the previous session (pid 2200056) was a genuinely wedged Happy wrapper — 0% CPU, up 3.7 days, sleeping — still holding the ~#1345 registration and worktree but doing nothing"; pod idle since 2026-07-18 07:27Z, recovered only after Thomas's "are you sure it's not running still?".

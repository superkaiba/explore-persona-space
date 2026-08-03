---
title: 'daily-fix: poller pid-identity check + pgrep memory sync'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e36d4b5ddca4
- daily-auto-filed
created_at: '2026-07-24T06:47:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): poll_pipeline has no pid-identity
  check so a fresh-but-wrong pid file is not runtime-detectable, and the experimenter
  pgrep memory still prescribes the acquisition path clause 1d banned'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-23 parked-candidate routing pass (Step C). Raised as TWO recursion-guard-parked prose follow-ups on task #1634 (Alternatives critic, 17:19Z) — both children of #1634's clause-1d pid-file work ("pid file from launch chain child, never pgrep"), grouped here because they share one subject (pid acquisition integrity).

## Goal

(1) Add a poller-side pid-identity check to `scripts/poll_pipeline.py` so a fresh-but-wrong pid file is runtime-detectable; (2) reconcile the experimenter agent-memory `feedback_pgrep_self_match_pidfile.md` with the new clause 1d so it no longer prescribes the partially-superseded pgrep-based relaunch pid resolution.

## Workflow gap

- **Bug observed:** (a) `scripts/poll_pipeline.py` has no pattern-probe fallback / pid-identity check — a fresh-but-wrong pid file (the stale-pid false-"run exited" class; two false monitor alarms fired today before pid capture was hardened in #1634) is not runtime-detectable; #1634's plan §9 explicitly deferred this poller-side mechanical extension (diagnosis-shortening, not correctness-critical). (b) `.claude/agent-memory/experimenter/feedback_pgrep_self_match_pidfile.md` prescribes pgrep-based relaunch pid resolution (self-match hygiene only) — partially superseded by #1634's clause 1d (pgrep acquisition ban + launch-chain-child pid capture).
- **Why it is a workflow gap:** a stale memory steering the experimenter toward the banned acquisition path undoes the landed #1634 fix; the poller-side identity check closes the runtime-detection half the same plan deferred.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "pid.identity\|pattern.probe" scripts/poll_pipeline.py` → 0 hits (absence claim, in-target 0-hit is the evidence); `.claude/agent-memory/experimenter/feedback_pgrep_self_match_pidfile.md` exists (1,636 B, unmodified since Jun 12 — predates clause 1d) and `grep -n "1d\|acquisition ban"` in it → 0 hits (2026-07-24 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Poller: on breadcrumb read, verify the pid's identity (cmdline matches the expected launch signature) before trusting liveness; fall back to a bracketed pattern probe on mismatch. Memory: rewrite the pgrep memory to point at clause 1d's launch-chain-child capture as the acquisition path, keeping pgrep for self-match hygiene only.

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`, `.claude/agent-memory/experimenter/feedback_pgrep_self_match_pidfile.md`

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: e36d4b5ddca4

- workflow_fix_target: scripts/poll_pipeline.py

Origin: parked candidates on #1634 (2026-07-23T17:19:04Z, items 1 + 2).

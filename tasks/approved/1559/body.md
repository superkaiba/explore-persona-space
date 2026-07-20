---
title: 'daily-fix: tmux guard - exempt dead single-socket rm'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b12d150a3d8b
- daily-auto-filed
created_at: '2026-07-20T06:47:04Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): single-file rm of verified-dead
  tmux socket blocked as broad sweep'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from a transcript-mined problem (session 0da7071f @ 18:48 UTC, interactive tmux-cleanup follow-up).

## Goal

Refine `.claude/hooks/guard_tmp_tmux_sweep.sh` so a SINGLE-FILE `rm` of a tmux socket whose server is provably dead (no live process holds it) is exempt from the broad-/tmp-deletion-sweep block, or explicitly document the verify-then-`EPM_ALLOW_TMP_SWEEP=1` flow as the intended path.

## Workflow gap

- **Bug observed:** the hook BLOCKED `rm -f /tmp/tmux-1001/old` (one verified-dead socket file) as a "broad /tmp deletion sweep" false positive; the agent had to fall back to the `EPM_ALLOW_TMP_SWEEP=1` override.
- **Why it is a workflow gap:** a single-file socket rm after a liveness check is not a sweep; the block adds an override round-trip to a safe, common cleanup. (Counterpoint the planner should weigh: the override path exists precisely to force verify-then-override — the emitter itself rates this "arguably working-as-designed".)
- **Confidence (emitter):** low (filed per the standing any-confidence directive; a reasoned no-change deflection is a fine outcome)
- verified-at-filing: `ls .claude/hooks/guard_tmp_tmux_sweep.sh` → exists (project-local hook). Transcript evidence (session 0da7071f, 2026-07-19 ~18:48 UTC): `BLOCKED: broad /tmp deletion sweep (rm target '/tmp/tmux-1001/old' ...)` followed by a successful `EPM_ALLOW_TMP_SWEEP=1` retry. Presence claim is transcript-anchored; no grep-refutable count claim made.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from the mined problem: exempt single-file rm of a provably-dead tmux socket, e.g. gate on "target is one file under /tmp/tmux-*/ AND no live process has it open")

## Scope / surfaces

- Primary target: `.claude/hooks/guard_tmp_tmux_sweep.sh`

## Constraints / invariants

- Workflow-surface only. The hook must stay fail-closed on anything resembling a multi-path or glob sweep.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: .claude/hooks/guard_tmp_tmux_sweep.sh
- fingerprint: e0b9cad082d6

Mined evidence (no candidate block was emitted in-session): PreToolUse block on `rm -f /tmp/tmux-1001/old`, self-corrected via the sanctioned override after re-verifying the server was dead.

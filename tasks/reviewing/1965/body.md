---
title: 'daily-fix: route long judge drivers off VM on 1st kill'
kind: infra
tags:
- wf-fix
- wf-fix-fp:08c3053254fc
- daily-auto-filed
created_at: '2026-08-01T07:05:33Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): A choom-protected multi-hour
  API-bound judge driver was earlyoom-collateral-killed in a fleet-wide storm (#1900);
  the SKILL recovery ladder licenses the CPU-pod pivot only after a SECOND protected
  kill, re-exposing hours of wall to the live storm (the CONSOLIDATED ''no memory-pressure
  watcher pass'' half is REFUTED — watcher pass #849 exists)'
workflow: v1
---
# daily-fix: route long judge drivers off VM on 1st kill

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED H3; miner-4:P2). Source session: 879efc0d (#1900) — the P2 full-judge VM driver (pid 2493177) was SIGKILLed by earlyoom at 2026-07-31T11:33Z during a fleet-wide low-memory storm; the session's own `epm:failure` marker (`assert_tag: earlyoom-collateral:issue1900_judge`) records "mem avail 9.85%, swap 0; earlyoom journal shows a python sweep, badness 966" — the driver's own RSS was small, the spiker was neighbor load, and `choom=-600` had been applied. Recovered from checkpoint via the #1019 machinery (4/6 batches had landed server-side).

## Goal

Default long VM-side API-bound judge drivers to the cheap CPU pod lane on the FIRST protected earlyoom collateral kill (instead of the current relaunch-once-then-pivot-on-the-SECOND-kill ladder).

## Workflow gap

- **Bug observed:** A choom-protected, multi-hour, API-bound judge driver was collateral-killed by earlyoom in a fleet-wide storm; under the current SKILL.md recovery ladder the sanctioned next step is ONE more VM relaunch with protection verified, re-exposing hours of checkpointed wall to the same live storm before the pod pivot is licensed. (Kill mechanism is the session's own marker diagnosis — probed by that session from the earlyoom journal, not re-derived here.)
- **Why it is a workflow gap:** SKILL.md's own text states choom "re-orders earlyoom's victim selection; it does not exempt the phase", yet the "Collateral-kill signature + second-kill pod pivot" block licenses the CPU-pod route only "if a PROTECTED phase is earlyoom-killed AGAIN". For a long API-bound judge driver (near-zero CPU/GPU need, trivially pod-portable, `cpu-mid` fits) the relaunch-once default has negative expected value during an attributed fleet-wide storm. NOTE — the CONSOLIDATED entry's other half ("no memory-pressure watcher pass exists") is REFUTED at compose time and is NOT part of this filing: `scripts/autonomous_session_watch.py` already carries the CPU/memory-pressure guard pass (task #849, escalate-only; MemAvailable floor + earlyoom-kill journal attribution).
- **Confidence (emitter):** low-medium (the second-kill ladder is a deliberate design; the planner should weigh relaunch cost vs pod-provision latency and may deflect with a reasoned no-change report)
- verified-at-filing: `grep -n -i 'earlyoom\|choom\|cpu-mid' .claude/skills/issue/SKILL.md` → context read at L6866-6877: "Recovery ladder: relaunch ONCE with protection verified (`choom=ok`); if a PROTECTED phase is earlyoom-killed AGAIN … route it to the cheap CPU pod lane (`cpu-mid` / `cpu-bigmem` by footprint)" — first-kill routing for API-bound drivers absent (0 hits for a driver-class or storm-conditioned first-kill clause). Refutation probe: `grep -n 'earlyoom\|MemAvailable\|choom' scripts/autonomous_session_watch.py` → 34 hits incl. "CPU/memory-pressure guard pass (task #849)" (L365-379, L4921+) — the watcher pass EXISTS. `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md scripts/autonomous_session_watch.py` eyeballed: no landed first-kill routing change (2026-08-01).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/skills/issue/SKILL.md § Collateral-kill signature + second-kill
pod pivot (~L6866):
- Recovery ladder: relaunch ONCE with protection verified (choom=ok);
- if a PROTECTED phase is earlyoom-killed AGAIN, … route to cpu pod lane
+ Recovery ladder: relaunch ONCE with protection verified (choom=ok) —
+ EXCEPT: a long API-bound driver (judge/batch-poll class: multi-hour,
+ checkpointed, no GPU need) killed during an ATTRIBUTED fleet-wide
+ storm (watcher #849 pressure row / earlyoom journal at the death
+ timestamp) routes to the cheap CPU pod lane (cpu-mid by footprint)
+ on the FIRST protected kill, resuming from its checkpoint. All other
+ phase classes keep the second-kill pivot unchanged.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (the recovery-ladder block).
- Grep the workflow surface for the pattern before editing (`grep -rn 'earlyoom-killed AGAIN\|second-kill' .claude/ CLAUDE.md`) — the gotchas.md earlyoom entry cross-references this ladder and must stay consistent.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff not applicable (markdown-only unless the planner extends).
- The refuted watcher-pass half must NOT be re-introduced — the #849 pass already exists; do not add a duplicate watcher lane.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- sha-verify (filing-time, #1467): `879efc0d` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 08c3053254fc

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: (driver-computed; tag authoritative)

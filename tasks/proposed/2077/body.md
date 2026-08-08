---
title: 'workflow-fix: wall-clock heartbeat duty for subagents owning detached work'
kind: infra
tags:
- wf-fix
- wf-fix-fp:399b0d367152
created_at: '2026-08-05T03:07:15Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation during #1739 follow-up, 2026-08-04: subagent
  jobb-ladder armed a session-scoped completion poll as its only liveness channel
  on a 4xH200 staging job; harness killed the poll (task bk2sldrt5, status killed),
  the stage process then died of EDQUOT at 23:58Z, no wake ever arrived, pod billed
  idle ~3h at $18.36/hr with the agent''s last marker 3h stale. The #1850 [watch-heartbeat]
  rule that covers this lives only in .claude/skills/issue/SKILL.md, which Agent-tool
  subagents never load.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a candidate raised during
task #1739 follow-up work (emitting agent: orchestrator, own observation).

## Goal

Add a wall-clock heartbeat duty for any agent — subagent included — owning long
detached work to CLAUDE.md § "Orchestrator vs subagent re-invocation", and state
that a session-scoped background poll shares its kill domain with the work it
watches, so poll silence is not evidence of health.

## Workflow gap

- **Bug observed:** an Agent-tool subagent owning a 4× H200 staging job
  (`pod-1739-r5ladder`) armed a session-scoped completion poll as its only
  liveness channel. The harness killed the poll (task `bk2sldrt5`, terminal
  status `killed`); the watched stage process then died of MooseFS EDQUOT at
  23:58Z; no wake ever arrived, so the subagent posted nothing after 22:20Z and
  the pod billed idle at $18.36/hr for ~3 h (~$55 of the round's ~$95) before
  the orchestrator noticed out-of-band.
- **Why it is a workflow gap:** the project HAS the exact rule that would have
  caught this — the `[watch-heartbeat]` Monitor-liveness duty (#1850: "a
  heartbeat gap of ≳2-3 expected intervals means the Monitor died … never assume
  it is still watching"), itself derived from a prior #1739 incident. But it
  lives ONLY in `.claude/skills/issue/SKILL.md` Step 6d.2, which an Agent-tool
  subagent never loads. Subagents DO load CLAUDE.md, and CLAUDE.md carries no
  subagent-side monitor-liveness or wall-clock heartbeat duty at all. The
  adjacent CLAUDE.md re-arm bullet (added 2026-08-04, commit `9817520b91`,
  #1947) covers the *wake-consumed-by-other-traffic* case — "re-arm as the FIRST
  action of every wake" — which by construction cannot fire when NO wake ever
  arrives. That is the uncovered case.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rln "watch-heartbeat" CLAUDE.md .claude/rules/ .claude/agents/ .claude/skills/` → 1 file, `.claude/skills/issue/SKILL.md` only (2026-08-04). Per-target: `CLAUDE.md` 0 hits; `.claude/rules/crash-fix-rounds.md` 0 hits. `grep -cn "watch-heartbeat\|Monitor died\|heartbeat gap" CLAUDE.md` → 0. Absence-of-guard claim, so the 0-hit in-target result IS the evidence (§ verified-at-filing clause (a)); the semantic-probe clause (a') is satisfied by the three-fragment repo-wide grep above rather than a single verbatim literal. Landed-fix check: `git log --oneline --since='7 days ago' -- CLAUDE.md` → 8 commits, none covering subagent monitor-liveness (`9817520b91` is the wake-consumed sibling described above, not this case).

## Proposed change (candidate diff sketch — refine in planning)

In CLAUDE.md § "Orchestrator vs subagent re-invocation", immediately after the
existing "Re-arm a bg-Bash poll chain as the FIRST action of every wake" bullet:

```
+ - **A poll you armed is not a heartbeat: it shares your kill domain with the
+   work it watches.** Any agent — ORCHESTRATOR OR SUBAGENT — that owns detached
+   work outliving its current turn posts a durable progress marker on a WALL-CLOCK
+   cadence (≤ ~60 min), whether or not any poll, Monitor line, or notification
+   fires. A session-scoped background Bash/Monitor can be reaped by the harness
+   with no error and no notification, after which silence is indistinguishable
+   from health and the watched compute runs unobserved (and, on a pod, billing).
+   Treat a heartbeat gap of ≳2-3 expected intervals as evidence the watcher DIED:
+   re-probe the work directly (`ps`/`pgrep` identity match, log mtime, pod status)
+   before re-arming, per `.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch.
+   Never heartbeat blind — a verify FAIL routes to the failure path instead.
+   Subagent briefs for detached work RESTATE this cadence and name the terminate
+   deadline. (The `/issue` orchestrator's fuller form is SKILL.md Step 6d.2
+   `[long-phase-heartbeat]` + the #1850 `[watch-heartbeat]` Monitor emission; this
+   bullet is the CLAUDE.md-level duty that also reaches Agent-tool subagents,
+   which never load that SKILL.)
```

Planning should decide whether the duty also belongs in the `experimenter` /
`experiment-implementer` agent specs, or whether the CLAUDE.md bullet suffices
given those agents load it.

## Scope / surfaces

- Primary target: `CLAUDE.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rn "watch-heartbeat\|long-phase-heartbeat" CLAUDE.md .claude/`) and
  keep the new bullet consistent with the SKILL.md machinery it points at — the
  `[long-phase-heartbeat]` marker token is read by `scripts/tick_triage.py`
  (`LONG_PHASE_HEARTBEAT_PREFIX`) and `scripts/autonomous_session_watch.py`, so do
  NOT mint a new token; reuse the existing one if the bullet prescribes a marker
  shape at all.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Do not weaken the existing re-arm bullet — this is additive and covers the
  disjoint no-wake-ever case.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: CLAUDE.md
- fingerprint: 399b0d367152

<!-- workflow-fix-candidate v1 -->
target_file: CLAUDE.md
bug_observed: a subagent owning a 4xH200 staging job armed a session-scoped completion poll; the harness killed the poll, no wake ever arrived, and the pod billed idle ~3h while the agent last posted a marker 3h earlier
why_workflow_gap: the equivalent duty (#1850 `[watch-heartbeat]` Monitor-liveness) exists only in .claude/skills/issue/SKILL.md, which Agent-tool subagents never load; CLAUDE.md has zero subagent-side monitor-liveness duty, and its adjacent re-arm-on-wake bullet cannot fire when no wake ever arrives
proposed_change: add a wall-clock heartbeat duty for any agent (subagent included) owning long detached work to CLAUDE.md section Orchestrator vs subagent re-invocation, and state that a session-scoped background poll shares its kill domain with the work it watches so poll silence is not evidence of health
diff_sketch: |
  + - **A poll you armed is not a heartbeat: it shares your kill domain with the
  +   work it watches.** Any agent — ORCHESTRATOR OR SUBAGENT — owning detached
  +   work that outlives its turn posts a durable progress marker on a WALL-CLOCK
  +   cadence (≤ ~60 min), whether or not any poll or notification fires. Treat a
  +   gap of ≳2-3 expected intervals as evidence the watcher DIED; re-probe the
  +   work directly before re-arming. Never heartbeat blind.
confidence: high
related_task: #1739
<!-- /workflow-fix-candidate -->

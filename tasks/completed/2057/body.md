---
title: 'workflow-fix: re-arm bg poll chain first on every wake (message consumes tick
  wake)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fc1df9c4dc27
created_at: '2026-08-04T02:09:28Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed on #1947: both concurrent lanes'' backend_poll
  chains died silently within ~2h; owning agent forensics confirmed a completed tick''s
  wake was consumed by an intervening teammate-message turn that ended without re-arming.
  Two 8-GPU queued jobs left unwatched.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a defect hit live on task #1947 (emitting agent: orchestrator, confirmed by `result3-theory-1947`'s own forensics).

## Goal

Document the re-arm-first rule: a subagent maintaining a background-Bash poll chain re-arms as the FIRST action of every wake, regardless of what woke it, until the watched job reaches a terminal state.

## Workflow gap

- **Bug observed:** an incoming teammate or orchestrator message consumes the wake that a completed background-Bash poll tick would otherwise trigger; if that turn ends without re-arming, the poll chain dies silently with no error and long-running compute runs unwatched.
- **Why it is a workflow gap:** the background-Bash poll chain is the SANCTIONED long-wait pattern (CLAUDE.md § "Orchestrator vs subagent re-invocation", the "End the turn when bg work is in flight" bullet at line 100, which explicitly bans foreground sleep chains and bg sleep-loops and names bg-Bash polling as the correct shape). But the pattern is only self-sustaining if every wake re-arms, and the rule as written never says so. The wake budget is shared: a message delivered to the subagent occupies the same wake slot the tick completion would have used, so ROUTINE ORCHESTRATOR TRAFFIC silently disarms the very watch the rule prescribes. There is no error, no marker, and no signal — the only symptom is `pgrep` returning nothing much later.
- **Confidence (emitter):** high — hit both concurrent lanes on one issue within ~2 h, and the owning agent's forensics independently confirmed the mechanism.
- verified-at-filing: `grep -n -iE 're-arm|rearm|consume.*wake|wake.*consume' CLAUDE.md .claude/rules/*.md` → **0 hits in CLAUDE.md** (8 hits total, all in `.claude/rules/{compute-backend-failover,background-automation,pod-side-reporting}.md` and all about unrelated re-arming: a PIPE re-arm, watcher retry-budget re-arms, and a phase-log rotation before re-arming a grep watcher — none about subagent wake consumption). Section anchors confirmed present: `### Orchestrator vs subagent re-invocation` at CLAUDE.md:94, the bg-poll bullet at CLAUDE.md:100 (2026-08-04 UTC). So the hosting section exists and the rule is absent from it.

**Live evidence (task #1947, 2026-08-03/04).** Two concurrent inline GPU rounds each armed a background-Bash poll against their own suffixed handle. Roughly two hours later `pgrep -af 'backend_pol[l].py'` from the orchestrator session returned NOTHING — both chains dead, both 8-GPU queued jobs unwatched. `result3-theory-1947`'s forensics: "the prior tick had completed cleanly (JSON: status running / current_phase pending, tick_rc=0) — its completion wake was consumed by an intervening teammate-message turn and I failed to re-arm in that turn." The orchestrator's own coordination messages (lane assignment, panel corrections) were the consuming traffic. Both lanes failed the same way independently, which is what makes this a pattern defect rather than one agent's slip.

## Proposed change (candidate diff sketch — refine in planning)

Add to CLAUDE.md § "Orchestrator vs subagent re-invocation", adjacent to the existing bg-poll bullet:

```
+- **A bg-Bash poll chain must be RE-ARMED AS THE FIRST ACTION OF EVERY WAKE,
+  regardless of what woke you, until the watched job reaches a terminal state
+  (COMPLETED / FAILED / CANCELLED / TIMEOUT).** The wake slot is SHARED: an
+  incoming SendMessage occupies the same wake the tick completion would have
+  used, so a turn that handles a message and ends without re-arming kills the
+  chain silently — no error, no marker, and the watched compute runs
+  unobserved. Treat "have I re-armed?" as the first question of the turn, not
+  the last. Size the cadence to the expected wait (a deep-queued job wants
+  ~1800s, not ~540s): a short cadence multiplies wake count and therefore
+  multiplies the chance that one wake is consumed without re-arming.
+  (#1947, 2026-08-04: two concurrent lanes each lost their poll chain to
+  orchestrator coordination traffic within ~2 h; both 8-GPU jobs sat queued
+  and unwatched until a manual `pgrep` found it.)
```

**Note for planning:** consider whether the ORCHESTRATOR side also warrants a line — when messaging a subagent known to be maintaining a poll chain, expect to have consumed a tick and say so in the message. That is a weaker, advisory-grade remedy; the subagent-side re-arm-first rule is the load-bearing one and should not depend on orchestrator discipline.

**Also worth considering in the same plan:** whether a mechanical backstop is warranted for long-running inline rounds — e.g. the autonomous-session watcher noticing an issue with a live `epm:run-launched`, a non-terminal SLURM job, and no live poller, and escalating. That is a larger change than the doc fix and should not block it.

## Scope / surfaces

- Primary target: `CLAUDE.md` (§ "Orchestrator vs subagent re-invocation", near line 100)
- Possibly also `.claude/rules/pod-side-reporting.md` if the poller-lifecycle guidance there should cross-reference it.
- No code change required for the documentation fix; the optional watcher backstop would touch `scripts/autonomous_session_watch.py` and should be scoped separately if the planner judges it in scope.

## Constraints / invariants

- Workflow-surface only — no experiment code, `configs/`, or `tasks/`.
- Must not contradict the existing bans in the same bullet (no foreground sleep chains; no bg sleep-loop watchers; no `nohup ... &` nested inside a bg-Bash whose exit is the completion signal).
- `scripts/workflow_lint.py` passes; if CLAUDE.md changes, `--check-lessons-index` stays consistent.
- This session runs under the recursion guard and does NOT auto-file further candidates.

## Provenance

- workflow_fix_target: CLAUDE.md
- fingerprint: fc1df9c4dc27

Surfaced from task #1947 while running two concurrent user-directed inline GPU rounds. Immediate mitigation applied by hand: both agents were messaged to re-arm and instructed to make re-arming the first action of every wake; cadences lengthened to 1800s. That mitigation is per-agent and verbal — it does not survive into future rounds, which is why the rule belongs in the always-on surface.

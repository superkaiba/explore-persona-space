---
title: 'workflow-fix: no-progress tick wedge — heartbeats read HEALTHY, stale-redrive
  reuses exhausted context'
kind: infra
tags:
- wf-fix
created_at: '2026-08-04T03:34:23Z'
has_clean_result: false
origin_prompt: why did it get wedged and can we stop this from happening gain?
workflow: v1
---
## Overview / Motivation

An autonomous `/issue` session burned 2h16m in a no-progress loop that BOTH watcher lanes are structurally blind to. Observed live on #2054, 2026-08-04T01:12Z-03:28Z; recovered only by manual force-respawn.

## Goal

Detect a session that wakes, performs no durable work, posts a heartbeat, and sleeps — repeatedly — and force-RESPAWN it (fresh context) instead of re-driving it into the same exhausted context.

## Workflow gap

- **Bug observed:** #2054's orchestrator emitted four consecutive tick heartbeats (01:52, 02:20, 02:52, 03:20) whose own text reported no state change ("Chain state unchanged since heartbeat #2"; "unchanged. Composing Unit A.1 brief; no state change"). Zero commits, zero status changes, zero non-heartbeat markers across 2h16m. No lane escalated.
- **Why it is a workflow gap (two independent blind spots):**
  1. `scripts/tick_triage.py::compute_issue_verdict` decides liveness on marker AGE ONLY — `if marker_age_s is not None and marker_age_s <= stale_after_s: return ("HEALTHY", ...)`. A content-free heartbeat resets the clock exactly like real progress. There is no notion of PROGRESS in the predicate.
  2. When staleness DID trip (heartbeats ~30 min apart vs `STALE_S_DEFAULT = 25*60`), the verdict is `STALE-REDRIVE`, which re-drives the SAME session. If the root cause is that session's own context exhaustion, re-driving is futile by construction — it re-attempts the same failing turn. Only a fresh session clears it.
  3. The watcher's existing force-respawn (prompt-wedge) lane keys on `isApiErrorMessage: true` assistant rows. These turns were NOT api errors — they completed successfully and did nothing. So that lane never fires.
- **Root cause of the underlying stall (worth encoding separately):** the orchestrator chose to compose a subagent brief with FULLY-INLINED code bodies, explicitly to defeat subagent autocompact thrash — heartbeat #2: "fully-inlined code required to survive micro-scoped subagent thrash". That inverts the standing rule in CLAUDE.md § 429 token-pacing ("Keep subagent prompts lean — pass the PATH to the plan/brief, never inline the body") and exhausted the ORCHESTRATOR's context instead. It self-diagnosed the correct fix at heartbeat #3 ("dispatch with a lean brief pointing at parent references rather than fully-inlined code") and could not execute it — the signature of context exhaustion, where cheap short turns (heartbeats) still succeed but the long composition turn never completes.
- **Confidence:** high
- verified-at-filing: `grep -n "marker_age_s <= stale_after_s" scripts/tick_triage.py` -> 1 hit at the HEALTHY branch of `compute_issue_verdict` (~L974), confirming the age-only predicate; `sed -n '950,978p' scripts/tick_triage.py` read in full, no progress term present; `STALE_S_DEFAULT = 25 * 60` at L113. #2054 heartbeat texts quoted verbatim from `task.py view 2054 --json` events at 01:52:12Z / 02:20:19Z / 02:52:09Z / 03:20:06Z (2026-08-04).

## Proposed change (refine in planning)

1. Add a PROGRESS term to the tick verdict. Track, per issue, a durable progress fingerprint — e.g. (latest non-heartbeat marker ts, HEAD sha on the issue branch, status). If the fingerprint is UNCHANGED across N consecutive ticks (suggest N=3) while markers keep arriving, return a new verdict `NO-PROGRESS-RESPAWN` rather than HEALTHY or STALE-REDRIVE.
2. Route that verdict to a force-RESPAWN (stop + `spawn-issue --auto`, preserving the session's auto-approve cap), not a re-drive — the whole point is dropping the poisoned context. Bound it the way the other wedge lanes are bounded: per-issue per-UTC-day cap, episode dedup, kill switch.
3. Consider making heartbeat markers self-declaring (a structured `progress: none` field) so the predicate does not have to string-match note prose. A heartbeat that admits no progress is the cleanest possible signal and the current skill already writes those words.
4. Separately: add an explicit anti-pattern to the lean-brief rule — "do NOT inline code bodies into a subagent brief to defend against subagent thrash; that converts a subagent context problem into an orchestrator context problem. Pass paths; if a subagent thrashes, micro-scope its work instead."

## Scope / surfaces

- Primary: `scripts/tick_triage.py` (the verdict predicate), `scripts/autonomous_session_watch.py` (the respawn lane).
- Secondary: `.claude/skills/issue-tick/SKILL.md` (heartbeat shape / structured no-progress field).
- Secondary: `CLAUDE.md` § 429 token-pacing (the lean-brief anti-pattern in item 4).

## Constraints / invariants

- Do not weaken the existing HEALTHY path for genuinely-working sessions that post infrequent markers — the progress fingerprint must key on durable evidence (commits / status / non-heartbeat markers), not on marker cadence.
- Respawn must be bounded and kill-switched, matching the conventions of the existing wedge lanes.
- `scripts/workflow_lint.py` passes; mapped tests pass.

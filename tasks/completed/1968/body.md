---
title: 'daily-fix: hold subagent respawn past active 429 storm'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2c4a2980a6d9
- daily-auto-filed
created_at: '2026-08-01T07:06:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): During the 2026-08-01 ~04:35-04:53Z
  output-TPM storm an orchestrator re-spawned its 429-killed analyzer ~2 min later
  mid-storm and the retry died within a minute (#1895); both CLAUDE.md 429 bullets
  license immediate retry with no storm-liveness check'
workflow: v1
---
# daily-fix: hold subagent respawn past active 429 storm

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED M1; miner-2:P12, miner-3:P13, miner-4:P12, miner-5:P3, miner-6:P4, miner-7:P11). Source sessions: the org-wide output-TPM 429 storm ~04:35-04:53Z (2026-08-01) hit ≥6 autonomous sessions ("Request rejected (429) … 3,000,000 output tokens per minute (model: claude-fable-5)"). The one actionable defect (db531944, #1895 — miner-2, probed row counts): the analyzer r2 spawn was 429-killed, the orchestrator re-spawned the SAME brief at 04:40 while the storm was live, and the retry died the same way within a minute; the transcript ends on a raw 429 with the task at `interpreting`.

## Goal

Amend the 429 re-spawn guidance so a 429-killed subagent is re-spawned only after the ACTIVE storm has passed, not merely "briefly" / at the next minute boundary.

## Workflow gap

- **Bug observed:** During a live multi-minute output-TPM storm, "wait briefly" was read as ~2 min and the identical subagent brief was re-spawned into the still-hot storm, burning the spawn and leaving the session dead on a raw 429 (probed row counts per miner-2; the stop-hook and watcher lanes recovered most OTHER sessions).
- **Why it is a workflow gap:** Both guidance surfaces license an immediate retry with no storm-liveness check: the user-global CLAUDE.md § Sub-agent rate limits says "Wait briefly, then re-spawn the same sub-agent with the same prompt", and the project CLAUDE.md § Context hygiene "429 token-pacing" bullet says "On a 429: wait for the next minute boundary and retry the same call". The hook (`~/.claude/hooks/on-stop-429-retry.sh`) is storm-AWARE on its own lane (per-session storm counter, cap 5 consecutive re-wakes, minute-boundary + jitter pacing) — but the ORCHESTRATOR-side re-spawn guidance has no "is the storm still live?" predicate, so a fleet-wide multi-minute storm defeats the minute-boundary heuristic.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n -A4 'Sub-agent rate limits' /home/thomasjiralerspong/.claude/CLAUDE.md` → L24-28, text "Wait briefly, then re-spawn the same sub-agent with the same prompt" (no storm-hold clause — presence hit context read, current text confirmed); `grep -n '429 token-pacing' CLAUDE.md` → L148, text "On a 429: wait for the next minute boundary and retry the same call" (no storm-hold clause); `grep -n -i 'storm' /home/thomasjiralerspong/.claude/hooks/on-stop-429-retry.sh` → storm counter + 5-cap present (hook-side aware; orchestrator-side guidance absent). `n/a` on git-log for the two user-global files (outside this repo); `git log --oneline --since='7 days ago' -- CLAUDE.md` eyeballed: no landed 429-respawn change (2026-08-01).

## Proposed change (candidate diff sketch — refine in planning)

```
/home/thomasjiralerspong/.claude/CLAUDE.md § Sub-agent rate limits:
- Wait briefly, then re-spawn the same sub-agent with the same prompt.
+ Before re-spawning, check the storm is OVER: if the stop-hook is still
+ cycling retries, or any 429 landed in the last ~2 min, the storm is
+ live — HOLD (poll each minute) until one clean minute passes, then
+ re-spawn at a minute boundary + jitter with the same prompt.
CLAUDE.md (repo) § Context hygiene, 429 token-pacing bullet:
+ mirror clause: on a 429 inside an ACTIVE storm (repeat 429s within
+ ~2 min), hold for a clean minute before the boundary-retry.
```

## Scope / surfaces

- Primary target: `/home/thomasjiralerspong/.claude/CLAUDE.md` (§ Sub-agent rate limits — NOTE: user-global, OUTSIDE this repo's working tree; the spawned session edits it in place, no Step 10d merge carries it); secondary: repo `CLAUDE.md` (§ Context hygiene 429 bullet). The hook itself needs no code change (its storm cap already binds its own lane).
- Grep the workflow surface for the pattern before editing (`grep -rn '429' CLAUDE.md .claude/ /home/thomasjiralerspong/.claude/CLAUDE.md`) and keep both bullets consistent with the hook's documented pacing.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes on the repo-side edit.
- The hold must stay bounded (never an indefinite wait — cap the hold at ~10-15 min then proceed per the existing retry rule) so a permanently-elevated baseline can't wedge a session.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 2c4a2980a6d9

- workflow_fix_target: /home/thomasjiralerspong/.claude/CLAUDE.md
- fingerprint: (driver-computed; tag authoritative)

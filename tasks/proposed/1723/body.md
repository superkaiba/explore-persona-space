---
title: 'daily-fix: CRON-TEARDOWN precedes the Step 10d merge, leavin'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4279aa5705ab
- daily-auto-filed
created_at: '2026-07-27T07:16:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): Step 10 tears down the
  issue-tick backstop before the Step 10d merge, so a 33-minute merge with conflict
  recovery ran with no re-drive coverage, and a task was marked completed with epm:done
  83 min before its merge landed'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 2 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Move the Step 10 CRON-TEARDOWN and the `completed` / `epm:done` transition to AFTER the Step 10d merge posts `epm:merged`, so the merge window keeps re-drive coverage and the durable record never claims done while the branch is unmerged.

## Workflow gap

- **Bug observed:** Step 10 step 6 tears down the `/issue-tick` backstop cron and applies the terminal `completed` status before Step 10d runs, so the entire merge — up to 33 minutes with conflict recovery and two background lint-gate waits — executes with no re-drive coverage and with `epm:done` already posted against an unmerged branch.
- **Why it is a workflow gap:** the ordering is written into `.claude/skills/issue/SKILL.md` Step 10 step 6, Step 10d is absent from the § CRON-TEARDOWN exit-site list, and two sibling sessions read the spec two opposite ways within the same hour.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'Run CRON-TEARDOWN before applying the terminal status' .claude/skills/issue/SKILL.md` → 1 hit (L9320, presence confirmed); `awk 'NR>=9711 && NR<=12320 && /CRON-TEARDOWN/{c++} END{print c+0}' .claude/skills/issue/SKILL.md` → **0** occurrences of CRON-TEARDOWN anywhere inside Step 10d (L9711–12320, the whole auto-merge step) — the absence-of-guard evidence; `grep -n 'at \`completed\` (Step 10 auto-complete)' .claude/skills/issue/SKILL.md` → 1 hit (L4693) confirming the exit-site list names Step 10 but not Step 10d; `grep -n 'long-phase-heartbeat' .claude/skills/issue/SKILL.md` → 2 hits (L4808, L6364), neither inside Step 10d (2026-07-26)

## Evidence

- Session `7ce3a81f`: `08:41:13.376Z CronList {}` → `08:41:17.400Z CronDelete {"id": "07ebadfd"}` → `08:41:20.282Z RESULT: No scheduled jobs.`, then Step 10d began at `[08:41:52] TOOL_USE Bash: … Guard 0: agent-memory pre-commit …` and the merge landed `MERGED mergedAt=2026-07-26T09:13:46Z`. The measured cost is 33 minutes of merge — including a shape-2 merge-conflict recovery and two roughly 12-minute background lint-gate waits — with no cron armed. No loss was realized on this run; the session survived.
- Sibling session `#1695` the same day deferred CRON-TEARDOWN to 09:25:24Z, after the merge landed — the opposite ordering, applied within 11 minutes of `7ce3a81f`'s. The spec is being read two ways.
- Session `e3b70618` (task #1709): `epm:done` posted at `2026-07-26T14:48:35Z`, `epm:merged` at `2026-07-26T16:11:00Z` — 83 minutes during which the durable record read `completed` + done with the branch unmerged. The `epm:merged` note reads `**Merge attempts:** 3 (2 blocked by workflow-lint gate; the third PASS came after dropping the …`.
- Session `8380a48c` shows the same shape at smaller scale: `epm:done` 10:16:41Z → `epm:merged` 10:51:09Z, a 34-minute window.
- That window is exactly the state the watcher's `completed_unmerged` pass (#1564/#1653) flags and, on a second flagged interval with no live owner, respawns `/issue <N> --auto` against. The pass runs hourly with a day-capped respawn — far slower than the 45-minute tick the teardown just removed. Its recovery row is already documented at `.claude/skills/issue/SKILL.md` L12458. On both 2026-07-26 sessions the owning session was alive and did merge, so no spurious respawn was observed; the ordering nonetheless makes a long-tail merge indistinguishable from a stranded one.
- unverified hypothesis — verify at plan time: that a session wedge or refusal-death inside the uncovered merge window would in fact have left the task `completed` + unmerged. No such death occurred in either transcript; the claim is inferred from the ordering, not observed.

## Proposed change

- In `.claude/skills/issue/SKILL.md` Step 10 step 6 (L9320): move CRON-TEARDOWN out of the pre-terminal-status position. Keep the `/issue-tick` backstop armed through Step 10d and tear it down at the Step 10d exit site, once `epm:merged` has been posted (or once Step 10d has terminally failed and routed to its `epm:merge-failed` handling).
- Reorder Step 10 so the terminal `completed` status and the `epm:done` marker (step 7) follow the Step 10d merge rather than precede it, so `epm:done` and `epm:merged` cannot be separated by a long merge window. If the reorder is judged too invasive for the `awaiting_promotion` experiment path (which merges at Step 9b and reaches Step 10 already merged), scope the reorder to the code-change path that actually merges at Step 10d.
- Add Step 10d to the § CRON-TEARDOWN exit-site list at `.claude/skills/issue/SKILL.md` ~L4691–4700, next to the existing `at \`completed\` (Step 10 auto-complete)` bullet, and update the parallel prose at ~L4839.
- Cheaper companion, and a useful backstop even if the reorder lands: on any Step 10d gate `block` or re-run, post an `epm:progress` `[long-phase-heartbeat] step10d-merge attempt=<k>` note, so the `completed_unmerged` audit in `scripts/autonomous_session_watch.py` can distinguish "merge in flight, attempt k" from "stranded". Alternatively have that pass treat an `epm:progress` newer than `epm:done` as liveness.
- Keep the teardown idempotent and keep the not-found-is-success semantics; the existing § CRON-TEARDOWN procedure is unchanged, only its call site moves.
- Add a contract pin so the two orderings cannot diverge again across sibling sessions.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- `scripts/autonomous_session_watch.py` (only if the `completed_unmerged` liveness-signal companion is taken)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- sha-verify (filing-time, #1467): `7ce3a81f` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `07ebadfd` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `e3b70618` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `8380a48c` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 4279aa5705ab

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: I-P2, J-P11.

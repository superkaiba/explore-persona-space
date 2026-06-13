---
title: 'Watcher infra-drain pass: execute the PM-curated dispatch queue (~/.eps-autonomous/infra-drain-queue.json)
  under cap 3'
kind: infra
tags: []
created_at: '2026-06-12T23:57:49Z'
has_clean_result: false
origin_prompt: Run it in background with ahppy coder
---
## Goal

Add a durable, pure-Python "infra-drain" pass to the 10-minute watcher (`scripts/autonomous_session_watch.py`) that executes the PM session's already-adjudicated infra dispatch queue, so ripe `proposed` `kind: infra`/`batch` tasks drain into free slots (cap 3) even when the PM session is closed.

## Why

The standing infra auto-dispatch rule (`.claude/agents/research-pm.md` § Standing rule, user directive 2026-06-12) only fires on PM-session STATUS passes. The PM session is event-driven, so cap-held ripe infra tasks sit undispatched whenever it is idle or closed. The user wants truly hands-off draining. A session-scoped hourly cron exists as a stopgap (dies with the PM session, 7-day expiry); this task is the durable replacement.

## Design (starting contract — refine as needed)

- The queue file ALREADY EXISTS at `~/.eps-autonomous/infra-drain-queue.json` (written by the PM session 2026-06-12). Schema: `ripe_oldest_first` (list of ints), `cap` (int, default 3), `holds` (dict id→reason; watcher must NEVER dispatch a held id), `updated_ts`, `updated_by`, `comment`. Schema refinements are fine — document them in the file's comment semantics and in research-pm.md.
- HARD CONSTRAINT: the watcher is pure Python with no LLM judgment. It must NEVER classify ripeness itself — it only executes IDs the PM session has already adjudicated into the queue file. Missing/empty/invalid file → the pass does nothing (log + skip).
- Pass logic:
  - occupied = count of tasks with kind in (infra, batch) at ACTIVE statuses (planning, plan_pending, approved, running, verifying, interpreting, reviewing), read via `explore_persona_space.task_workflow` (never hand-built `tasks/<status>/<N>` paths).
  - free = max(0, cap − occupied); dispatch up to `free` IDs from `ripe_oldest_first`, in order.
  - Per-ID guards: skip if task status != `proposed`; skip if `~/.eps-autonomous/issue-<N>.json` exists (already spawned — dedupes against the PM session and its hourly cron, which also drains at :23); skip if ID appears in `holds`; one spawn attempt per ID per watcher cycle, with a last-attempt timestamp so a failed spawn is never retried in a tight loop (backoff or attempt cap).
  - Spawn via the same mechanism the crash-recovery pass uses (`spawn_session.py spawn-issue --issue <N> --auto`).
- Follow the watcher's existing pass structure, logging, and env-var-override conventions; add a kill switch (e.g. `EPM_DISABLE_INFRA_DRAIN=1`) and document the new pass in `.claude/rules/background-automation.md`.
- Update `.claude/agents/research-pm.md` § Standing rule — infra auto-dispatch with 2–4 lines: the PM now WRITES the queue file on every STATUS pass (add/remove ripe IDs, update holds); the watcher executes between passes; the PM remains the only ripeness judge.
- Do NOT touch task.py status enums, marker schemas, or spawn_session.py CLI contracts.

## Acceptance criteria

1. Watcher pass implemented with all per-ID guards above; missing file = no-op.
2. Tests (`tests/test_workflow*.py` family) pinning: held IDs never dispatch; non-`proposed` IDs never dispatch; existing `issue-<N>.json` blocks dispatch; cap arithmetic; missing/invalid file = no-op; kill switch honored.
3. `workflow_lint.py --check-asks` passes; ruff clean on touched files (touched files only).
4. `.claude/rules/background-automation.md` + `research-pm.md` updated and consistent with the implementation.

## Provenance

Filed from the PM session. Originating user prompt (verbatim): "Run it in background with ahppy coder" — re-routing the same brief previously dispatched to an in-session workflow-improver subagent (stopped before it made any commits) into a dedicated Happy session. Candidate block logged in `.claude/cache/workflow-fix-events.jsonl` (2026-06-12T22:40:30Z).

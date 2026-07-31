---
title: 'daily-fix: set-status bounded index.lock retry before raise'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9a66ab399e6c
- daily-auto-filed
created_at: '2026-07-31T06:55:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): task.py set-status crashed
  mid-mutation on a transient index.lock collision (git exit 128), leaving a half-applied
  status move (#1815) needing hand-reconcile; the deliberate #898 fail-loud raise
  has no bounded lock retry in front of it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 (problem sweep; miner-2 P4, session c0d17512 / issue #1815).

## Goal

Stop a transient `.git/index.lock` collision from leaving `task.py set-status` in a half-applied state the session must hand-reconcile — via a bounded lock retry before the deliberate fail-loud raise, or (if the raise must stay untouched) a documented manual-reconcile recipe.

## Workflow gap

- **Bug observed:** `set-status 1815 planning` moved the folder on disk, then its internal commit raised `CalledProcessError` (git exit 128, index.lock collision), leaving `MM tasks/proposed/1815/events.jsonl` + untracked `tasks/planning/1815/`; the session hand-reconciled (~2 min + two guard blocks en route). Sibling paths (`new_plan_version`, `post_event`) deferred gracefully the same day.
- **Why it is a workflow gap:** under fleet concurrency index.lock collisions are routine (5 firings across 4 sessions on 07-30); a status move that half-applies on a TRANSIENT collision converts a known-benign contention event into manual repair.
- **Confidence (emitter):** medium
- **Deliberate-design context (compose-time read):** `src/explore_persona_space/task_workflow.py` ~L4630 documents "set_status is deliberately NOT converted to deferred behavior (#898 raise + ghost-sweep semantics stay) ... the raise IS the fail-loud contract" — the DEFERRAL half of the proposal is explicitly refuted by design. The residual gap is narrower: a bounded index.lock retry (the ~60s bounded-poll form CLAUDE.md prescribes for sessions) BEFORE the raise would not weaken the fail-loud contract for genuine failures, and/or the manual-reconcile recipe belongs in the /issue SKILL.
- verified-at-filing: `sed -n '4625,4635p' src/explore_persona_space/task_workflow.py` → the deliberate-raise comment quoted above is present (2026-07-31 filing time); the planner should treat "no bounded lock retry precedes the set-status commit" as an unverified hypothesis — verify at plan time against `_run_git`/`_git_commit`'s existing retry behavior.

## Proposed change (candidate diff sketch — refine in planning)

Either (a) wrap the set-status commit's git invocation in the bounded index.lock poll (retry once + wait up to ~60s for the lock to clear) before letting the #898 raise fire, or (b) planner deflects (a) with a reasoned no-change and instead documents the half-applied-state manual-reconcile recipe in `.claude/skills/issue/SKILL.md`.

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py` (set_status commit path); fallback doc target `.claude/skills/issue/SKILL.md`.

## Constraints / invariants

- The #898 fail-loud raise + ghost-sweep semantics MUST survive any change; a genuine (non-lock) git failure still raises.
- `tests/test_task_workflow*.py` invariants stay green; add a lock-retry pin test if (a).

## Provenance

- fingerprint: 9a66ab399e6c

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- origin: /daily 2026-07-30 miner-2 P4 (transcript c0d17512, #1815)

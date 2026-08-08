---
title: 'workflow-fix: register project-level -lean twin agents so the v20 autocompact-thrash
  remedy actually works'
kind: infra
tags:
- wf-fix
- wf-fix-fp:03381f25b9a0
created_at: '2026-08-04T14:48:04Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate raised from /issue 2061 Step 5b fail-loud:
  the v20 orchestrator-steer named code-reviewer-lean / experiment-implementer-lean
  / planner-lean / critic-lean / consistency-checker-lean as the autocompact-thrash
  remedy, but the Agent tool refuses them as unknown types in a fresh Happy session
  even though the files landed on origin/main (a242c9a03b, 2026-08-04 08:42 -0400).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #2061 (emitting agent: /issue skill orchestrator).

## Goal

Diagnose and fix the agent-type registration gap that prevents project-level `.claude/agents/*-lean.md` files (added by task #2062 / commit `a242c9a03b` on origin/main) from resolving as Agent-tool `subagent_type` values in a fresh Happy session spawned AFTER they landed, so the v20 orchestrator-steer's autocompact-thrash remedy actually works.

## Workflow gap

- **Bug observed:** on task #2061, both the standard `code-reviewer` spawn and its one bounded re-spawn (SKILL.md Step 5b item 4) autocompact-thrashed with no durable verdict, so I attempted the SKILL.md § Step 5b Autocompact-thrash respawn recipe's fallback `code-reviewer-lean`. The Agent tool refused it with "Agent type 'code-reviewer-lean' not found. Available agents: … analyzer-lean, …" — a stale registration set that includes only the USER-GLOBAL `~/.claude/agents/analyzer-lean.md` and OMITS every project-level `-lean` file on origin/main (`code-reviewer-lean.md`, `critic-lean.md`, `consistency-checker-lean.md`, `experiment-implementer-lean.md`, `implementer-lean.md`, `planner-lean.md`). Files verifiably present at `origin/main:.claude/agents/`; landed by `a242c9a03b` (2026-08-04 08:42 -0400 = 12:42Z, ~5 h BEFORE this /issue 2061 session was spawned at 14:06Z). Task #2061 is at `status: blocked` with `epm:failure v1 failure_class: infra` recording the incident.
- **Why it is a workflow gap:** the v20 orchestrator-steer (task #2061 events.jsonl v20, posted by the predecessor) explicitly names `code-reviewer-lean` / `experiment-implementer-lean` / `planner-lean` / `critic-lean` / `consistency-checker-lean` as the "newly-registered lean twins" — the sanctioned remedy for the autocompact-thrash class (SKILL.md Step 5b Autocompact-thrash respawn recipe). If the Agent tool cannot resolve them in a session where the on-disk files exist on `origin/main`, the SKILL.md remedy is inert and any experiment/infra task can get catastrophically blocked when its `code-reviewer` / `experiment-implementer` spawns thrash on the fixed-overhead ceiling. #2061 is that catastrophe realized; the pattern is fleet-wide because the fixed overhead (project CLAUDE.md 222 KB + spec 137 KB + MCP + brief) is the same for every task on this VM.
- **Confidence (emitter):** medium — the immediate symptom is clear (Agent tool refusal listing the resolved set), but the root cause (session-start caching vs a required user-global placement vs some other agent-registration mechanism) is unverified. The spawned `/issue --auto` session's planner + critics + code-reviewer are the second check.
- verified-at-filing: `git -C ~/explore-persona-space ls-tree -r --name-only origin/main -- .claude/agents/ | grep lean` → 6 project-level lean files present on origin/main (`code-reviewer-lean.md`, `consistency-checker-lean.md`, `critic-lean.md`, `experiment-implementer-lean.md`, `implementer-lean.md`, `planner-lean.md`); `ls ~/.claude/agents/*-lean.md` → 1 user-global file (`analyzer-lean.md`); Agent tool refusal on `subagent_type: code-reviewer-lean` returned an "Available agents" list including only `analyzer-lean` from the -lean family (2026-08-04T14:41Z).

## Proposed change (candidate diff sketch — refine in planning)

Investigate + fix. Likely paths in priority order:

1. **Diagnose the registration mechanism.** Determine whether project-level `.claude/agents/*.md` files are supposed to resolve as agent types in the Agent tool for a fresh Happy session AFTER their commit lands on main, and (if not) whether the mechanism is: (a) user-global placement required, (b) session-start cache with a staleness window, (c) requires `spawn_session.py spawn-*` to seed the agent list, (d) something else. The `~/.claude/agents/analyzer-lean.md` is the only member of the family that resolves — that is the load-bearing empirical difference.
2. **If path (a) — user-global placement required:** land parallel copies of the 6 project-level lean twin specs at `~/.claude/agents/{code-reviewer,critic,consistency-checker,experiment-implementer,implementer,planner}-lean.md`. Keep the project-level copies as the source of truth; make the user-global copies thin `Read the project file` shims OR symlinks. Add a workflow_lint check that flags drift between the two.
3. **If path (b) — session-start cache:** add a documented "restart your Happy session to pick up new agent types" note in `.claude/agents/` READMEs and to the v20 steer template in SKILL.md, so agents that hit the thrash class know to restart before assuming the lean twins are unavailable. AND: add a session-startup probe that logs the resolved-agent-type set on stdout (visible via `happy-ls` progress) so debugging future gaps takes seconds not hours.
4. **If path (c) — spawn seeding:** thread the current `.claude/agents/*.md` set into `spawn_session.py spawn-issue/spawn-pm/spawn-campaign` so new sessions inherit the up-to-date agent list.
5. **Post-fix verification:** on the new session that lands the fix, spawn each of the 6 lean twins with a trivial "read your spec and echo back" brief, confirm each resolves, and record the six spawn ids in the workflow-fix task's completion body.

## Scope / surfaces

- Primary target(s): `.claude/agents/*-lean.md` on origin/main (registration side), `~/.claude/agents/` (if user-global placement is required), `.claude/skills/issue/SKILL.md` (v20 steer template + Step 5b Autocompact-thrash respawn recipe), and possibly `scripts/spawn_session.py` (if spawn-side seeding is the mechanism). Grep the workflow surface for `analyzer-lean` / `-lean` references to identify all sites needing updates.
- Grep the workflow surface for the pattern before editing: `grep -rln -E '(-lean|analyzer-lean)' .claude/ CLAUDE.md scripts/` and update every hit.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md § Recursion guard).
- Task #2061 remains at `status: blocked` until this fix lands OR the user manually resolves its code-review step; the fix does NOT retroactively unblock #2061 automatically (a fresh `/issue 2061` re-invocation after this fix lands is the resume path).

## Provenance

- workflow_fix_target: .claude/agents/*-lean.md, .claude/skills/issue/SKILL.md, ~/.claude/agents/, scripts/spawn_session.py
- fingerprint: 03381f25b9a0

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/*-lean.md, .claude/skills/issue/SKILL.md, ~/.claude/agents/, scripts/spawn_session.py
bug_observed: Standard code-reviewer and experiment-implementer subagents both autocompact-thrash on this project's fixed-overhead ceiling (project CLAUDE.md 222 KB + agent spec 137 KB + MCP schemas + brief), leaving the subagent context with too little headroom for the 81 KB diff + 15 KB plan reads. Task #2061 got blocked at Step 5 code-review after both the initial spawn and the one bounded re-spawn (per SKILL.md Step 5b item 4) thrashed with no durable verdict.
why_workflow_gap: The v20 orchestrator-steer names `code-reviewer-lean` and 4 sibling lean twins as the sanctioned autocompact-thrash remedy. On disk the files exist on `origin/main` (landed by #2062 / a242c9a03b at 2026-08-04 08:42 -0400, ~5 h before this session), but the Agent tool's session-loaded set does not include them — only `analyzer-lean` (from `~/.claude/agents/`) resolves. The SKILL.md remedy is therefore inert.
proposed_change: Add the `-lean` twin agent files (code-reviewer-lean, experiment-implementer-lean, planner-lean, critic-lean, consistency-checker-lean) that the v20 orchestrator-steer names as the autocompact-thrash remedy but which are not yet registered as agent types (only analyzer-lean exists today).
diff_sketch: |
  # Diagnose + one of:
  # (a) Land parallel user-global copies at ~/.claude/agents/*-lean.md (shim or symlink)
  # (b) Add a session-startup probe + a "restart to pick up new agent types" note
  # (c) Thread agent-list into spawn_session.py so new sessions inherit up-to-date types
  # Verify by spawning each lean twin with a trivial echo brief; record ids.
confidence: medium
related_task: #2061
<!-- /workflow-fix-candidate -->

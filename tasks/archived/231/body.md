---
title: 'Refactor parallel dispatch to use Claude Code agent teams (Wave 4 of #202)'
kind: infra
tags: []
created_at: '2026-05-04T20:03:25.000Z'
has_clean_result: false
sagan_id: 1f4c68cc-216b-4214-a07a-bd21b0ae61e8
sagan_number: 231
priority: normal
legacy_why_unset: true
---
## Goal

Refactor `/issue` and `/adversarial-planner` to use Claude Code's native agent-team primitives (`TeamCreate`, `TeamDelete`, `Agent()` with `team_name`) for phases that benefit from parallel dispatch.

**Parent:** #202 (Workflow optimizations — Waves 1-3 landed in PR #217)

## Background

This is Wave 4 / PR-H from the #202 plan. It was intentionally deferred because:
1. Agent teams are experimental (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`)
2. It's the highest-risk item (touches core orchestrator)
3. It depends on Waves 1-3 landing first (especially PR-E parallel critics and PR-F/G SKILL.md flow changes)

## Design decisions (from #202 plan, approved)

1. **Teams only for parallel phases.** Sequential phases (planner, experimenter, reviewer) keep direct `Agent()` calls. Teams add value only where 2+ agents run concurrently.
2. **Marker comments remain primary state.** Teams are ephemeral (die with the session); markers are durable (survive across sessions). No migration of state from markers to team context.
3. **Teams created per invocation, not persistent.** Since teams don't survive session boundaries, each `/issue <N>` invocation creates a fresh team if needed for the current phase.
4. **`isolation: "worktree"` replaces manual `git worktree add`** for implementer dispatch in Step 4a.

## Phases that benefit from teams

| Phase | Current | With teams |
|-------|---------|-----------|
| `/adversarial-planner` Phase 2 | 3 parallel `Agent()` critics (from PR-E) | 3 critic teammates in a team |
| `/issue` Step 9a | Sequential analyzer → interpretation-critic | Potentially parallelizable |

## Fallback design (required)

If the teams API is too limited (idle/wake management too complex, limitations block us):
- Fall back to 3 `Agent()` calls with `run_in_background: true`
- Feature flag: `USE_AGENT_TEAMS=true` in the skill so it can be reverted without touching the rest of the flow

## Known API limitations (from docs)

- No session resumption with in-process teammates
- Task status can lag
- No nested teams
- One team per session
- Lead is fixed
- Teammates go idle after every turn (need explicit SendMessage to wake)

## Acceptance criteria

- [ ] `/adversarial-planner` Phase 2 runs 3 critics via a team (or fallback) on a test task
- [ ] `/issue` Step 4a uses `isolation: "worktree"` instead of manual `git worktree add`
- [ ] Feature flag allows reverting to non-team dispatch
- [ ] Dry-run on a synthetic issue completes full lifecycle
- [ ] All existing tests pass
- [ ] Documented in SKILL.md with the new flow

## Compute

0 GPU-hours. Pure infra work.

## Estimate

8-12 hours (multi-day). Develop in a worktree.

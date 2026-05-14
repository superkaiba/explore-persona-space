---
title: Refactor to use agent teams in a worktree
kind: infra
tags: []
created_at: '2026-05-01T18:46:01.000Z'
has_clean_result: false
sagan_id: 0f2946dc-a2d5-48b7-beab-80e1d1dd5283
sagan_number: 172
priority: normal
---
**Goal.** Refactor the multi-agent workflow (`/issue`, `/adversarial-planner`, and all other multi-agent skills) to use Claude Code's native agent-team primitives (`team_name` + `isolation: "worktree"` on `Agent` calls; `TeamCreate` / `TeamDelete` for lifecycle) in place of the current ad-hoc agent spawning + manual `git worktree add`.

**Why now.** The current workflow predates Claude Code's team primitives. Manual worktree creation, ad-hoc agent spawning, and the implicit "marker comments as state" protocol all reinvent capabilities that the harness now provides natively. Aligning with the native primitives should simplify orchestration, reduce moving parts, and give the agents proper isolation by default.

**Scope.** Full redesign of the multi-agent workflow.
- All multi-agent orchestrator skills are in scope: `/issue`, `/adversarial-planner`, the iterative interpretation loop, and any other place we currently spawn multiple coordinated agents (e.g., `/auto-experiment-runner`, `/clean-results`, `/experiment-proposer`).
- Agent definitions in `.claude/agents/*.md` may need updates to fit the new team grouping.

**Constraints.**
- Develop the new workflow inside a git worktree (e.g., `.claude/worktrees/issue-172/`) so `main` stays clean while we experiment.
- This is a **hard cutover** — the refactor replaces the current workflow rather than running alongside it. The old code path can be deleted once the new one is proven.
- Backward-compatibility for in-flight issues is explicitly NOT required (we will land this when no experiment is mid-pipeline).

**Out of scope.**
- The agent personas / role definitions themselves (gate-keeper voice, reviewer adversarial behavior, etc.) — those are unchanged; only how they are dispatched and how state flows between them changes.
- Pod / preflight / upload-verification logic — these stay as-is.

**Acceptance criteria.**
- The new `/issue` (or its replacement) runs an end-to-end issue lifecycle using `Agent({ team_name, isolation: "worktree", … })` instead of `git worktree add` + bare `Agent()`.
- All other multi-agent skills follow the same pattern.
- Documented in `.claude/skills/<…>/SKILL.md` with the new flow.
- A successful end-to-end dry run on a synthetic / trivial issue.

**Open design questions** (planner to resolve):
- Where does durable state live now? Marker comments on the issue, the team's persistent context, or both?
- Does the `Agent` tool's `team_name` survive across invocations (i.e., can `/issue` resume by re-attaching to an existing team), or does each `/issue` invocation create a fresh team?
- Exact mapping of the current `status:*` state machine onto team lifecycle events (`TeamCreate` / individual `Agent` runs / `TeamDelete`).

## Spec (from clarifier)

1. **Mechanism:** Use Claude Code's native primitives (`team_name`, `isolation: "worktree"`, `TeamCreate`/`TeamDelete`).
2. **Scope:** All multi-agent skills.
3. **Backward-compatibility:** Hard cutover — replaces the current workflow.
4. **Worktree:** Implementation lives in a worktree of this repo (the standard `/issue` worktree mechanism).

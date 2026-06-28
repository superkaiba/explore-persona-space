---
title: Route workflow-surface fixes through a background /issue --auto session (replace
  the workflow-improver auto-spawn) for full plan+critic review
kind: infra
tags: []
created_at: '2026-06-27T01:33:59Z'
has_clean_result: false
origin_prompt: I think somewhere it says to use workflow improver or implementer to
  make workflow fixes? Instead change it so it uses a background happy coder session
  (so we get the plan review and all this)
---
## Overview / Motivation

Workflow-surface fixes (changes to `.claude/agents/*.md`, `.claude/skills/**/SKILL.md`,
`.claude/rules/*.md`, `.claude/workflow.yaml`, `CLAUDE.md`, and the `scripts/`
orchestration helpers) currently **auto-spawn the `workflow-improver` subagent** (via the
`Agent` tool, in a worktree) paired with a SINGLE `code-reviewer`. That path has **no
planning stage and no adversarial-critic rounds** — it is the lightweight "edit + one
review" loop defined in `.claude/rules/workflow-fix-on-bug.md`. The workflow surface is
exactly where a bad change has the widest blast radius, so it deserves the heaviest
review, not the lightest.

User directive (Thomas, 2026-06-26): **workflow fixes should instead be handled by filing
a task + spawning a background `/issue <N> --auto` Happy session**, so they get the FULL
pipeline — planner → adversarial-planner critic rounds → implementer → code-reviewer
(Claude+Codex ensemble) → test-verdict → auto-complete (+ the plan-approval gate) — i.e.
"the plan review and all this", the same rigor experiments get.

## Goal

Change the **workflow-fix-on-bug protocol** so that when a workflow-fix candidate is
raised (a `<!-- workflow-fix-candidate v1 -->` block OR a surfaced prose follow-up), the
orchestrator's default action is to **file a `kind: infra` task** (pre-filled from the
candidate: target file(s), bug, proposed change, verbatim origin) and **spawn a
background `/issue <N> --auto` session** to implement it — REPLACING the current
`Agent(subagent_type="workflow-improver", ...)` auto-spawn. The fix then lands via the
`/issue` code-change path (planner/critic/code-reviewer/test/auto-complete + the Step 10d
worktree merge), not via workflow-improver-as-subagent.

## Design questions for the planner (decide + justify each)

1. **All candidates, or a threshold?** A full session per candidate is heavier than the
   current background subagent. Decide whether EVERY in-scope candidate routes to a
   session or whether genuinely-trivial one-liners keep a lighter path. User intent is
   "we get the plan review" → bias toward the full session for anything non-trivial.
   State the rule explicitly.
2. **Fate of `workflow-improver`.** Retire the agent entirely, OR repurpose it as the
   *implementer* the `/issue` session uses for the workflow-surface implement step (so
   its in-scope knowledge is reused under the full pipeline). Pick one; make every
   reference consistent.
3. **Candidate → task mapping.** Exactly how the orchestrator turns a candidate block /
   prose follow-up into `task.py new --kind infra --title ... --body-file ...
   --origin-prompt` + `spawn_session.py spawn-issue --issue <N> --auto`, including
   **dedup** (a candidate whose `target_file` already has an open workflow-fix
   task/session must not double-file) and carrying over the existing
   one-formal-block-per-invocation + grep-the-surface-first rules.
4. **Recursion guard.** A workflow-fix `/issue` session must NOT itself auto-file MORE
   workflow-fix sessions for its own work. The current rule's "subagents never spawn
   workflow-improver" + the `AUTO_REVIEW_DISABLED` guard must carry over as an analogous
   no-fan-out guard. Define it.
5. **Auto-merge / push reconciliation.** Today the orchestrator merges the
   workflow-improver worktree + pushes on return. Under the new path the `/issue` session
   owns its Step 10d worktree merge. Reconcile the standing "workflow-surface edits are
   committed + merged + pushed automatically, no approval gate" rule with the `/issue`
   plan-approval gate — architectural / public-contract workflow changes should still
   surface to the user, and the plan gate is the natural place.
6. **Markers.** Keep / adapt `epm:workflow-fix-candidate|applied|failed`
   (`workflow.yaml` § markers) so the dashboard still surfaces the lifecycle, now
   pointing at the spawned task.

## Scope / surfaces to update (grep for the full set FIRST)

Run `grep -rln 'workflow-improver' .claude/ CLAUDE.md scripts/` **excluding**
`.claude/worktrees/`, `.claude/cache/`, and `tasks/`, and update every canonical hit.
Known canonical surfaces:
- `.claude/rules/workflow-fix-on-bug.md` (the protocol — primary rewrite)
- `CLAUDE.md` (§ "Workflow-fix-on-bug protocol" summary)
- `.claude/agents/workflow-improver.md` (retire or repurpose per Q2)
- `.claude/agents/experiment-implementer.md` (reference)
- `.claude/skills/issue/SKILL.md` + `.claude/skills/issue/markers.md` (orchestrator spawn site)
- `.claude/skills/weekly/SKILL.md`, `.claude/skills/daily/SKILL.md` (mentions)
- `.claude/workflow.yaml` (§ markers)
- `.claude/rules/agents-vs-skills.md` (the agent ontology table)
- `scripts/workflow_lint.py`, `scripts/daily_surface_hook.sh` (if they reference it)

## Constraints / invariants (do NOT break)

- **Emission side unchanged:** subagents still emit `<!-- workflow-fix-candidate v1 -->`
  blocks / prose follow-ups; they NEVER spawn anything themselves. Only the orchestrator
  routes — now via task-file + session, not `Agent(workflow-improver)`.
- **Non-architectural in-scope fixes still default to AUTO** (no greenlight) — but "auto"
  now means "auto-file + auto-spawn `--auto` session"; its plan auto-approves under the
  GPU-h cap (these are ~0 GPU-h) yet still runs the full planner/critic/code-review.
- **Architectural / public-contract workflow changes still need the user's greenlight** —
  now expressed as the `/issue` plan-approval gate (park at `plan_pending`).
- `AUTO_REVIEW_DISABLED` / no-recursion semantics preserved (Q4).
- The out-of-scope set (experiment code, `configs/`, `tasks/`) is unchanged — never routes here.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if
  `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.

## Acceptance criteria

- `.claude/rules/workflow-fix-on-bug.md` + CLAUDE.md describe the task-file + background
  `/issue --auto` mechanism as the default; NO stale instruction to `Agent(...)`-spawn
  `workflow-improver` remains anywhere (grep-clean except a deliberate retire/repurpose note).
- The orchestrator-side recipe (SKILL.md) shows the exact `task.py new --kind infra ...`
  + `spawn-issue --issue <N> --auto` sequence, with dedup + recursion guard.
- The plan includes a dry walkthrough: candidate raised → task filed → session spawned →
  planner/critic/code-review → merged → markers posted.
- Tests + lint green.

## Context / pointers

- Current protocol: `.claude/rules/workflow-fix-on-bug.md` (auto-spawn workflow-improver,
  background, worktree; orchestrator merges + pushes on return).
- **This task is itself the FIRST instance of the new pattern, filed + spawned by hand**
  (bootstrap — the new mechanism can't route its own creation).
- Sibling background-session pattern: `scripts/spawn_session.py spawn-issue --auto`
  (used for #676 the same day).

---
title: 'Tighter auto-continuation: agent-spec audit + AskUserQuestion lint rule +
  CLAUDE.md sharpening'
kind: infra
tags:
- workflow
- lint
created_at: '2026-05-20T20:33:00Z'
has_clean_result: false
legacy_why_unset: true
---
# Tighter auto-continuation: agent-spec audit + AskUserQuestion lint rule + CLAUDE.md sharpening

## Why this experiment

**Application:** infra — serves Audit (forces agents to commit to a halt-criterion model the user can inspect) and Defend (no surprise pauses from automation interrupt the user's flow on the phone).

**Decision this changes:** Whether to trust the per-issue session topology to run end-to-end without user interruption. If the audit finds many stray `AskUserQuestion` calls outside the documented gates, the per-issue session is unsafe to dispatch unattended. If few/none, the lint rule alone suffices as forward defense.

**Expected outcome + branches:** I expect most agent/skill files have either no `AskUserQuestion` or only at-gate uses, with 1-3 stray cases in older agents that pre-date the gate enumeration. If the audit reveals >10 stray cases, that's a signal the agent doc culture has decayed and we need a stronger refactor (treat as a follow-up task). If 0-3 stray cases, just fix them inline and ship the lint rule.

**What gets cut if we run this:** Per-experiment dashboard scoping (item 4) does not run this week. Workflow primitives that govern agent behavior beat UX polish that surfaces the same data.

## Goal

Enforce the existing "no asking outside gates" policy mechanically. Today the rule lives in CLAUDE.md prose and is honored by convention. This task makes it auditable + lint-protected.

Three deliverables:
1. **Audit + cleanup** of every `AskUserQuestion` reference in `.claude/agents/**.md` and `.claude/skills/**/SKILL.md`. Each surviving reference must be at a documented gate (workflow.yaml § gates.inline / gates.park_and_wait / gates.conditional) or be a meta-discussion of when NOT to use it.
2. **Lint rule** that catches future violations. Extends `scripts/workflow_lint.py` (or new module if cleaner). Walks the same paths; fails on any `AskUserQuestion` mention that isn't either:
   - Annotated with `<!-- gate: <gate_id> -->` resolving to a real workflow.yaml gate, OR
   - Inside a fenced code block tagged as documentation (e.g. an example demonstrating misuse), AND that example has a `<!-- example: anti-pattern -->` marker.
3. **CLAUDE.md sharpening** — turn the existing "auto-continuation policy" prose into a tighter contract: outside the 6+1+1 gates, agents MUST post `epm:failure` + `status:blocked` and exit. Document the contract that fires the new lint.

## Three enforcement points

### 1. Agent-spec audit (one-shot cleanup)

For every match in `.claude/agents/**.md` and `.claude/skills/**/SKILL.md` of the literal token `AskUserQuestion`:

- **(a) Already at a documented gate** — add a `<!-- gate: <gate_id> -->` annotation on the same line or the line immediately above. `<gate_id>` is a dotted key resolving in workflow.yaml (e.g. `gates.inline.plan_approval`, `gates.inline.why_experiment`, `gates.park_and_wait.awaiting_promotion`).
- **(b) Outside any gate** — replace with the halt-criterion pattern:
  ```
  Post `epm:failure v1` with `failure_class: <code|infra|data>` and the
  specific blocker. Set status:blocked. Exit. The user re-invokes /issue <N>
  after reading the blocker.
  ```
  Remove the `AskUserQuestion` mention entirely; do NOT leave it as a comment.
- **(c) Meta-discussion** (e.g., "do NOT use AskUserQuestion outside gates") — annotate with `<!-- example: anti-pattern -->` so the lint can identify it.

### 2. Lint rule — `scripts/workflow_lint.py`

New `--check-asks` mode (added to the default pre-commit hook):

```
$ uv run python scripts/workflow_lint.py --check-asks
```

Walks `.claude/agents/**.md` and `.claude/skills/**/SKILL.md`. For each line containing the literal `AskUserQuestion`:

- **PASS** if line (or the line immediately above) contains `<!-- gate: <key> -->` AND `<key>` resolves to a real entry in `workflow.yaml § gates.{inline,park_and_wait,conditional}`.
- **PASS** if the line is inside a fenced code block AND that block (or the paragraph above it) has `<!-- example: anti-pattern -->`.
- **FAIL** otherwise, with `<file>:<line>` context and a pointer to this contract.

Exit code: 0 on PASS, 1 on FAIL (consistent with the existing `--check-references` / `--check-tables` modes).

### 3. CLAUDE.md sharpening

Augment the existing "Auto-continuation policy" section (lines ~36-83). Specifically:

- Add explicit halt-criterion behavior for "I would otherwise ask the user": "Outside the gates above, NEVER use `AskUserQuestion`. If your decision genuinely needs user input, post `epm:failure v1` with `failure_class: <code|infra|data>`, set `status:blocked`, and exit. The user re-invokes `/issue <N>` after reading the blocker."
- Add the lint reference: "Enforced by `scripts/workflow_lint.py --check-asks` (pre-commit hook)."
- Add the annotation convention: "Every legitimate `AskUserQuestion` call site in `.claude/agents/**.md` or `.claude/skills/**/SKILL.md` must carry an inline `<!-- gate: <dotted_key> -->` annotation resolving to a workflow.yaml gate. Anti-pattern examples carry `<!-- example: anti-pattern -->`."

## Files touched

- `scripts/workflow_lint.py` — add `--check-asks` mode + integration into `main` / argparse.
- `CLAUDE.md` — sharpen the "Auto-continuation policy" section with halt-criterion + lint reference + annotation convention.
- `.claude/agents/**.md` — audit cleanup; expect 5-15 files touched depending on audit findings.
- `.claude/skills/**/SKILL.md` — same.
- `.pre-commit-config.yaml` — add `--check-asks` to the workflow_lint hook invocation.
- `tests/test_workflow_lint.py` (new or extended) — coverage for the new lint mode.

## Test plan

Run all from repo root, post results in final report:

1. **Audit pass**: `uv run python scripts/workflow_lint.py --check-asks` PASSES after all cleanups (was FAILing or undefined before). Show the count of `AskUserQuestion` references found, broken down by (a)/(b)/(c).
2. **Lint coverage tests**: `tests/test_workflow_lint.py` covers (i) PASS on properly annotated, (ii) FAIL on unannotated, (iii) PASS on `<!-- example: anti-pattern -->` blocks, (iv) FAIL when gate annotation references a non-existent gate. Run via `uv run pytest tests/test_workflow_lint.py -v`.
3. **Regression — existing lint modes**: `uv run python scripts/workflow_lint.py --check-references` and `--check-tables` still PASS.
4. **Smoke regression — task workflow**: `uv run pytest tests/test_task_workflow.py tests/test_verify_task_body.py -v` still PASS (the round-2 fixture fixes from #371 must hold — DO NOT modify those fixtures).
5. **CLAUDE.md sharpening readable**: visually scan the diff; the auto-continuation section reads as a tighter contract.

## Acceptance criteria

- `--check-asks` integrates into the default pre-commit chain alongside `--check-references` and `--check-tables`.
- The audit reduces stray (b)-class `AskUserQuestion` mentions to zero.
- All legitimate (a)-class mentions carry a gate annotation that resolves.
- All (c)-class meta-discussions carry the anti-pattern annotation.
- Lint test coverage is direct (positive + negative cases for each match type).
- No regression in the existing lint modes or the task-workflow tests.

## Coordination with #371

#371 round 2 is fixing test fixtures + unbundling the Codex CLAUDE.md addition. Do NOT touch:
- `tests/test_verify_task_body.py` (round-2 fixture rewrites)
- `tests/test_task_workflow.py` (round-2 fixture rewrites)
- The `## Why this experiment` H2 section format
- `scripts/verify_task_body.py`
- `scripts/task.py` `_enforce_why_this_experiment_gate`
- `scripts/migrate_add_legacy_why_sentinel.py`

If #371 round 2 lands first, rebase on top. If this lands first, #371 rebases on top.

## Explicit cuts (not bundled)

- Adding `AskUserQuestion` reduction to subagents that aren't in `.claude/agents/**.md` (e.g., `general-purpose`, plugin-provided agents). Out of project scope.
- Replacing existing `<!-- gate: -->` annotations with a stricter machine-readable schema (e.g., YAML frontmatter on every gate spot). Defer.
- Per-experiment dashboard (item 4 of the workflow-changes queue).

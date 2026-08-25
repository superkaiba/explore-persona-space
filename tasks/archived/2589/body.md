---
title: 'workflow-fix: Codex review-site dispatches run write-enabled despite declaring
  read-only'
kind: infra
tags:
- wf-fix
created_at: '2026-08-25T21:06:52Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2587 orchestrator at /adversarial-planner Phase 2:
  the epm:codex-task-spawned marker recorded write=True while all three codex-critic
  composers declared ''Codex write mode: false (read-only critic)''; zero --no-write
  occurrences in the canonical review-site dispatch invocations.'
workflow: v1
---
# workflow-fix: Codex review-site dispatches run write-enabled despite declaring read-only

## Goal

Make the five doubled Codex review sites dispatch their reviewer with write access DISABLED, so the read-only intent the `codex-*` composers already declare is actually enforced rather than merely stated.

## The gap

`scripts/codex_task.py` defaults `--write` to TRUE:

- `scripts/codex_task.py:1715-1722` — mutually-exclusive `--write` / `--no-write` group, with `--write` help text reading "Grant Codex write access (default)."
- `scripts/codex_task.py:1907` — comment: "Default for --write is True (grant write) unless --no-write was passed."
- `scripts/codex_task.py:640` — appends `--write` to the companion command.

The canonical review-site dispatch invocation passes neither flag, so it silently takes the write default. Verified: zero occurrences of `--no-write` across the canonical invocations in `.claude/skills/adversarial-planner/SKILL.md`, `.claude/rules/codex-ensemble-review.md`, and `.claude/skills/issue/SKILL.md`.

Meanwhile every `codex-*` composer returns a dispatch config whose last line reads `Codex write mode: false (read-only critic)`. Observed live on task #2587 Phase 2: all three composers (`codex-critic` methodology / statistics / alternatives) declared `write mode: false`, and the resulting `epm:codex-task-spawned` marker recorded `write=True`. The declaration is documentation, not a constraint.

## Why it matters

A plan critic, an interpretation critic, a clean-result critic and a follow-up critic have no legitimate need to mutate the repo — their whole contract is to return a verdict marker. Granting write access to an agent whose spec says read-only is wrong on its own terms, and it carries concrete risk:

- Review sites routinely dispatch THREE Codex reviewers concurrently (one per lens at the `critic` site). Three write-enabled agents at the SHARED repo root, during heavy fleet activity, is the exact writer-concurrency class that `.claude/rules/repo-root-uncommitted-state.md` documents as destructive (the pre-commit stash race permanently loses writes that land inside another session's hook window).
- The artifact under review is itself a repo file (`plans/plan.md` / a task body / a clean-result body). A reviewer that can edit what it is grading breaks the independence the doubled-review design exists to provide.
- `code-reviewer` sites review a working-tree diff, so a write-enabled reviewer there could silently alter the very diff under review.

Note the helper's write default is CORRECT for its other caller: `/codex:rescue` needs write. The defect is not the helper default — it is that the REVIEW sites never opt out.

## Scope

- Add `--no-write` to the canonical Codex dispatch invocation at every review site: `.claude/rules/codex-ensemble-review.md` (the canonical bg-Bash form), `.claude/skills/adversarial-planner/SKILL.md` (Phase 2 / Step 4b dispatch + the implementation-pattern pseudocode), and `.claude/skills/issue/SKILL.md` (the Step 5 / 9a / 9b / follow-up-critic dispatch sites).
- Consider whether the five `codex-*` composer specs should emit the flag as part of their returned dispatch config, so the orchestrator copies a complete command rather than reconstructing one.
- Consider a mechanical guard: a `workflow_lint.py` check that any `codex_task.py` invocation appearing in a REVIEW-site surface carries `--no-write`, so the gap cannot silently reopen. `/codex:rescue` surfaces must stay exempt.
- Leave `scripts/codex_task.py`'s own default alone (rescue depends on it).

## Acceptance criteria

1. Every review-site `codex_task.py` invocation in `.claude/**` passes `--no-write`.
2. A fresh Codex review dispatch records `write=False` in its `epm:codex-task-spawned` marker.
3. `/codex:rescue` retains write access (no regression).
4. If a lint check is added, it FAILs on a review-site invocation missing `--no-write` and PASSes on a rescue invocation.

## Provenance

Surfaced by the #2587 orchestrator during `/adversarial-planner` Phase 2 (2026-08-25) while dispatching three Codex lens critics: the spawn marker's `write=True` contradicted all three composers' declared `write mode: false`, and grepping the canonical invocations found no `--no-write` anywhere. Not a #2587 experiment bug — a gap in the shared workflow surface affecting all five doubled review sites fleet-wide.

The #2587 round was allowed to continue on the already-dispatched write-enabled runs rather than being killed and re-dispatched: the prompts are critique-only and ask for a verdict marker, an actual write would be visible in git, and killing three in-flight high-effort runs costs real compute. That judgement is specific to one round and is not a reason to leave the gap open.

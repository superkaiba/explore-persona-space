---
title: Agent-memory index commits from a worktree silently drop rows and accumulate
  merge=union duplicates
kind: infra
tags:
- workflow-fix
created_at: '2026-08-27T21:44:09Z'
has_clean_result: false
origin_prompt: 'Found during /issue 2546 clean-result-critic gate: committing the
  Codex composer''s agent-memory write from the issue worktree would have dropped
  3 rows present on origin/main, and the worktree index carried 6 duplicate rows from
  the merge=union driver.'
workflow: v1
---
# Agent-memory index commits from a worktree silently drop rows and accumulate duplicates

## Goal

Close two defects in how `.claude/agent-memory/*/MEMORY.md` indices survive being written by a
subagent inside an issue worktree and then committed. Both were caught by hand on #2546; neither
is caught by any mechanical gate today.

## What happened (measured, #2546, 2026-08-27)

The `codex-clean-result-critic` composer wrote one new memory body plus an index row while
running in the `issue-2546` worktree. Before committing, the orchestrator ran the
gotchas.md no-lost-row `comm` check against `origin/main` and found the worktree index was
unfit to commit in two independent ways.

**Defect 1: stale worktree copy silently drops rows.** Three rows present at `origin/main`
were absent from the worktree's index, and their body files were absent from the branch too:

    feedback_open_interp_ids_at_cr_gate.md
    feedback_recipe_snapshot_spot_check.md
    feedback_reconciler_revise_verification_rounds.md

Committing the worktree copy as-is would have landed an index that drops all three. Because a
MEMORY.md row is a POINTER to a body file, a dropped row orphans a body silently: no conflict
fires, and the body file stays on disk looking healthy. That is exactly the #2093/#2101
incident shape (7 reconciler rows dropped undetected).

**Defect 2: the `merge=union` driver accumulates duplicates without bound.** The worktree
index carried 14 content rows of which 6 were duplicates:

    feedback_delta_rounds_beyond_r3.md          x2, PLUS a third differently-titled row
                                                for the same file with drifted summary text
    feedback_lens13_plan_fetch_patch.md         x2
    feedback_prior_round_prompt_reuse.md        x2
    feedback_fold_round_context_file_briefs.md  x2
    feedback_spec_inline_brace_false_positive.md x2

Union concatenates both sides rather than reconciling them, which is correct for never losing a
row and wrong for keeping the index readable. Nothing dedupes afterward, so these indices grow
and the same body accumulates two or three rows whose summaries drift apart. The drifted
delta-rounds pair is already mildly contradictory ("cap is 10 (was 5 under #1017)" vs "cap is
now 1-10 (was 1-5/#1017)").

Main's index turned out to be a strict superset of the worktree's apart from the one new row, so
the repair was to take main verbatim, append the new row, and restore the three bodies from main.
Landed as `603aea99aff` on `issue-2546`.

## Why the existing rule did not fire on its own

`.claude/rules/gotchas.md` carries the no-lost-row check, but its stated trigger is
"before REPLACING a local MEMORY.md with another copy (manual import, stale-copy alignment,
any `git checkout <ref> -- .claude/agent-memory/...`, or any operation whose effect is
'other copy wins')". Committing a stale worktree copy is not naturally read as any of those:
nothing is being replaced and no other copy is winning. The check ran on #2546 only because the
orchestrator happened to reach for it. Meanwhile `.claude/rules/repo-root-uncommitted-state.md`
names agent-memory as the fleet's dominant standing-armer class and tells the session to commit
those files in the same turn, which is a nudge toward committing FAST rather than toward
checking first.

## Proposed scope (for the planner to evaluate, not prescriptive)

1. **Extend the no-lost-row trigger** so it explicitly covers committing an agent-memory index
   from a worktree, naming `origin/main` as the comparison base, and reconcile the wording with
   the repo-root-uncommitted-state commit-in-the-same-turn duty so the two rules do not pull in
   opposite directions.

2. **Add a mechanical gate** so this stops depending on orchestrator recall. Candidate: a
   `scripts/workflow_lint.py --check-agent-memory-index` arm, bundled into the no-flags default
   run, that for every staged or committed `.claude/agent-memory/*/MEMORY.md`:
   - FAILs on two rows referencing the same body file (defect 2);
   - FAILs when a row present at `origin/main` is absent from the version under check (defect 1);
   - FAILs when a referenced body file is absent from the tree under check;
   - FAILs on an orphaned body file present on disk with no index row (the inverse leak).
   A pre-commit hook arm is the alternative placement; the lint arm is preferred because the
   Step 9c gate already runs it fleet-wide.

3. **Decide the `merge=union` question.** Either keep union and pair it with a dedupe step at
   the same place the gate runs, or replace it with a driver that reconciles. Keeping union
   unchanged plus a duplicate-FAIL gate is coherent only if something dedupes automatically;
   otherwise every future union merge hands a human a red gate to fix by hand.

## Acceptance criteria

- A stale-vs-main agent-memory index, a duplicate row, a row pointing at an absent body, and an
  orphaned body each FAIL a mechanical check with a message naming the offending file and row.
- Fixture tests reproduce all four shapes, including the exact #2546 worktree index as a
  regression fixture.
- The gotchas.md trigger wording covers the worktree-commit path, and the tension with the
  repo-root-uncommitted-state duty is resolved in the text rather than left implicit.
- The eleven `.claude/agent-memory/*/MEMORY.md` indices currently in the repo are swept once and
  reported: duplicates, stale rows, orphaned bodies. Existing duplicates are deduped in that
  sweep, or the sweep explains why each survives.

## Provenance

Found while #2546 committed a Codex composer's round-1 agent-memory write during the
clean-result-critic gate. The by-hand repair is `603aea99aff`; the both-directions `comm`
output and the full duplicate census are in #2546's events.jsonl at the marker following it.

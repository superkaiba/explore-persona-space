---
title: 'workflow-fix: canonical Codex review-dispatch snippet omits --no-write, so
  doubled review sites run write-enabled'
kind: infra
tags:
- wf-fix
created_at: '2026-08-25T18:10:56Z'
has_clean_result: false
origin_prompt: 'Discovered during /issue 2578 code-review round 1: the codex-code-reviewer
  composer declared ''Codex write mode: false (read-only review)'' but the canonical
  CLAUDE.md dispatch snippet omits --no-write and codex_task.py defaults to write=True,
  so epm:codex-task-spawned recorded write=True. .claude/agents/codex-code-reviewer.md:710
  prescribes --no-write; CLAUDE.md and .claude/rules/codex-ensemble-review.md do not.'
workflow: v1
---
# Canonical Codex review-dispatch snippet omits `--no-write`, so every doubled review site runs with file-write access

## Goal

Make the canonical Codex dispatch snippet for the five doubled review sites grant Codex read-only access, so a reviewer cannot mutate the tree it is reviewing. Today the canonical snippet omits `--no-write` while `scripts/codex_task.py` defaults to write-granted, so an orchestrator that copies the documented form dispatches every review round with write access.

## The contradiction

Three surfaces disagree about the same dispatch:

1. `CLAUDE.md` § "Codex ensemble review" — the canonical bg-Bash invocation. Omits `--no-write`.
2. `.claude/rules/codex-ensemble-review.md` — the same snippet, the file CLAUDE.md points to as authoritative for "before dispatching, composing, or posting any twin verdict". Omits `--no-write`.
3. `.claude/agents/codex-code-reviewer.md:710` — the agent spec's own dispatch snippet. **Passes `--no-write`.**

`scripts/codex_task.py` resolves the default at line ~1907: `write = True if args.write is None else args.write`, with `--write` help text reading "Grant Codex write access (default)." So omission is not neutral — it grants write.

An orchestrator following CLAUDE.md (the always-on surface) therefore contradicts the agent spec it is dispatching. Grep confirms nothing in `.claude/rules/` or `CLAUDE.md` prescribes `--no-write`; the only prescriptions live in the agent spec and in `.claude/agent-memory/codex-code-reviewer/feedback_revision_round_compose_recipe.md`, which discusses `--no-write` as an assumed condition ("`--no-write` Codex under the never-execute rule") rather than as an instruction to the dispatcher.

## Why it matters

The review sites are read-only by design — `codex-clean-result-critic.md` states the twin is "dispatched read-only", and the `codex-*` wrappers are prompt-composers that never execute. A write-enabled reviewer can:

- mutate the worktree it is reviewing, contaminating the round diff the ensemble is grading;
- leave dirt that trips the Step 10d merge guards and the Step 5a `#1972` uncommitted-dirt arm (which widens a family skip on any dirty member);
- write into the shared repo root, which is the `#2015` pre-commit stash-race hazard class.

Nothing has been observed writing yet. This is a latent fail-open in the dispatch contract, not a reported corruption.

## Observed on

Task #2578, code-review round 1 (2026-08-25). The `codex-code-reviewer` composer explicitly declared `Codex write mode: false (read-only review)` in its handoff, and the orchestrator then dispatched via the canonical CLAUDE.md snippet — producing `epm:codex-task-spawned` with `write=True`. The composer's declared intent and the realized dispatch disagreed, and no surface caught it. The orchestrator detected it only by reading `codex_task.py`'s argparse defaults after noticing the marker text.

## Acceptance criteria

1. The canonical snippet in `CLAUDE.md` § "Codex ensemble review" and in `.claude/rules/codex-ensemble-review.md` passes `--no-write` for all five doubled review sites, with a one-line note that the helper defaults to write-granted so omission is not neutral.
2. Decide and record ONE disposition for the helper default, and state the reasoning either way: (a) flip `codex_task.py`'s default to read-only so review dispatch is safe-by-default and write becomes opt-in for any genuine write task, or (b) keep the default and rely on the corrected snippets. Option (a) is the fail-safe direction and should be preferred unless a live caller genuinely needs write-by-default — enumerate the callers before deciding.
3. A mechanical pin so the surfaces cannot drift apart again: a `workflow_lint.py` check (or a `tests/test_*` prose pin) asserting the review-dispatch snippets in `CLAUDE.md` / `.claude/rules/codex-ensemble-review.md` / `.claude/agents/codex-*.md` agree on the write mode.
4. `--reattach` recovery snippets carry the same write mode as the dispatch they resume.

## Scope notes

Prose + one flag default + one pin. No GPU. Do not widen this into a redesign of the Codex dispatch path or the composer contract.

Estimated GPU-hours (total): 0

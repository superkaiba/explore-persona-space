---
title: 'workflow: raise ensemble review cap 3->5, expand git-provenance auto-strip,
  surface-not-ship real residual at cap'
kind: infra
tags:
- workflow-review-cap
created_at: '2026-07-01T05:56:55Z'
has_clean_result: false
origin_prompt: 'change to cap 5 and if it still hasnt converged then we just continue
  [refined after evidence: reconciler ledger ~58% of non-convergence = real blocker,
  so surgical version: cap 3->5 on the 4 iterating sites + expand auto-strip to git-provenance
  false-positive classes + surface (not ship, not pivot-loop) genuine real residuals
  at cap]'
---
## Overview / Motivation

User directive (2026-06-30): the ensemble code-review round cap of 3 is too low — it
triggers a strategy-pivot back to `planning` (heavy) when the two remaining blockers are
often small, patchable, 0-GPU-h fixes. Raise the cap and change the cap-hit behavior, but
do it SURGICALLY: the reconciler's own ledger shows ~58% of non-convergence-at-cap cases
resolve as a REAL blocker one reviewer under-flagged (55 claude-under-flagged vs 39
codex-overreach corrections), and the real classes are data-loss / wasted-compute /
wrong-result bugs (silent-failure, HF upload-path, missing checkpoint-per-phase, resource
leaks, unplumbed pipelines, producer/consumer key mismatches, missing smoke). So blind
"continue past FAIL" would ship past a real bug more often than not. Instead: give more
rounds to converge, strip the well-characterized FALSE-POSITIVE blocker classes so they
never count as non-convergence, and for a genuine real residual SURFACE it rather than
ship past it or pivot-loop.

## Goal

Change the ensemble-review cap-hit policy: raise the per-reviewer round cap 3 → 5 on the
four iterating review sites; expand the orchestrator auto-strip to cover the git-provenance
false-positive blocker classes; and replace the cap-hit strategy-pivot with
strip-false-positives-then-continue, surfacing (not shipping past, not pivot-looping) any
genuine real residual.

## Change set (grep the surface first; move every "cap 3" / "round cap 3" reference together)

1. **Round cap 3 → 5** for the four iterating ensemble review sites (critic, code-reviewer,
   interpretation-critic, clean-result-critic). Uniform; low-risk (more rounds to converge;
   the reconciler already resolves many splits before the cap). Update `.claude/workflow.yaml
   § ensemble_review`, the "Round cap 3 per reviewer" text in `CLAUDE.md`, `.claude/skills/
   issue/SKILL.md`, and any agent files that state "cap 3".

2. **Expand the orchestrator auto-strip** (currently strips mechanical-contract-only FAILs:
   `marker-shape`, `smoke-run-missing` — SKILL.md Step 5c-bis; code-reviewer.md Steps
   0.5/0.6/0.7) to ALSO strip the **git-provenance false-positive class**, gated on
   orchestrator verification that the flagged issue is genuinely NOT caused by the round's
   diff:
   - pre-existing-on-trunk failures (a test/lint already broken on main)
   - stale-main / stale-worktree diffs (flagging deletions/changes that are diff-base artifacts)
   - cumulative `main...HEAD` diffs (flagging prior rounds' already-reviewed changes)
   These are the recurring Codex-overreach classes in the reconciler ledger
   (`feedback_codex_litigates_pre_existing_in_round_n`, `feedback_697_branch_diff_against_stale_main`,
   `feedback_preexisting_test_breakage_sft`). Stripping them removes NON-REAL causes of
   non-convergence WITHOUT loosening the real gate. The strip must be evidence-based (the
   orchestrator verifies pre-existence via `git`), never a blanket ignore.

3. **Cap-hit behavior (replaces the cap-3 strategy-pivot):** after cap-5 without convergence,
   apply the strip (step 2), then:
   - if ALL residual blockers are stripped / false-positive → **PASS / continue**;
   - if a genuine REAL residual remains → **SURFACE it, do NOT ship past it and do NOT
     pivot-loop**: interactive → present to the user; autonomous → `epm:failure v1`
     (`failure_class: code`) + `status:blocked` + notify (the standing halt path for a
     genuinely-stuck real blocker after the pivot/patch space is exhausted).
   The real-vs-false-positive classification uses the two taxonomies above (real:
   silent-failure, upload-path/artifact-loss, checkpoint-per-phase, resource-leak,
   scaffolded-but-unplumbed, producer/consumer mismatch, missing/incomplete smoke,
   estimand/headline-poisoning; false-positive: the git-provenance class + the existing
   mechanical cosmetics).

## Rationale (why not blind "continue")

Empirically ~58% of non-convergence-at-cap = a real blocker (reconciler ledger 55:39), and
the real classes lose data / waste GPU / corrupt results. #779 is the exemplar: its residual
blockers were `raw-completions-upload-prefix-missing` (artifact-loss) and an incomplete
stage-0 smoke carve-out (code not proven to run) — blind-continue there burns the 8×H100 run.
Cap-5 gives two more patch rounds (real residuals are usually small/patchable and clear);
stripping the false-positive class removes the non-real churn; surfacing the rare genuine
stuck-real residual avoids both shipping a real bug and infinite pivot churn.

## Per-site nuance to preserve

- The expanded git-provenance strip is primarily a **code-review** concern (that's where those
  false positives arise); the cap-5 change is uniform across all four sites.
- The surface-real-residual rule matters MOST for the two post-run record gates
  (interpretation-critic, clean-result-critic): a real residual there is a flagged OVERCLAIM,
  which must be surfaced, never auto-published into the clean-result record.

## Constraints / invariants

- Workflow-surface only. Grep `.claude/ CLAUDE.md scripts/` for every "cap 3" / "round cap"
  / "Cap-3 pivot" / mechanical-contract-strip reference and move them together (do not leave a
  stale "3" anywhere — this is the #622 sibling-hit class).
- Keep the reconciler + PASS-vs-FAIL + FAIL+FAIL(overlap/disjoint) logic intact; this changes
  only the cap VALUE, the strip TAXONOMY, and the cap-HIT terminal behavior.
- Tests: update any test pinning `cap == 3`; add tests for the expanded strip taxonomy
  (git-provenance FP stripped only when git confirms pre-existence) and the new cap-hit
  behavior (strip→continue vs surface-real-residual). `scripts/workflow_lint.py --check-asks`
  passes; ruff on touched files passes; workflow.yaml ↔ CLAUDE.md ↔ SKILL.md stay consistent.
- Not architectural (parameter + policy change; no enum/schema/subcommand rename) → the /issue
  session may auto-approve + Step-10d auto-merge per the workflow-fix default.

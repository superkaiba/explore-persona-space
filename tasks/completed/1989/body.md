---
title: 'workflow-fix: repro-artifacts-clean gate vs working-tree git status'
kind: infra
tags:
- wf-fix
- wf-fix-fp:81d20e797f79
created_at: '2026-08-02T06:22:42Z'
has_clean_result: false
origin_prompt: 'clean-result-critic fold-round review on #1768, 2026-08-02 (candidate/prose
  follow-up; verbatim in body Provenance)'
workflow: v1
---
## Overview / Motivation
Auto-filed from a prose follow-up surfaced by clean-result-critic on task #1768 (fold-round review, 2026-08-02). The finding it would have caught mechanically: 16 untracked + 4 modified `operator_kv` result files sat in the working tree contradicting the parked body's key-side headline.
## Goal
Add a verify_task_body check that FAILs (or WARNs — planner decides severity vs grandfathering) when untracked or modified result files exist in the working tree under an `eval_results/issue_<N>/` subdir the body's `**Repro:**` footer names, for a task at `awaiting_promotion`.
## Workflow gap
- **Bug observed:** a post-fold uncommitted analysis pass (16 untracked + 4 modified files under eval_results/issue_1768/map_augmentation/operator_kv/) reversed the body's newest headline at layer 14; no gate compares body-claimed per-cell artifact counts/scope against working-tree git status, so the staleness was caught only by a fresh LM read.
- **Why it is a workflow gap:** the body-vs-artifact integrity lens has mechanical coverage for figures (check 29, tracked-at-HEAD, WARN) but nothing for RESULT files; the failure class generalizes to every awaiting_promotion task with post-park inline rounds.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -cn "porcelain\|untracked\|ls-files" scripts/verify_task_body.py` → 2 hits, neither a result-file gate (context read: check 29 covers body-LINKED FIGURES only, doc L548); absence of any eval_results working-tree comparison confirmed (2026-08-02).
## Proposed change (candidate diff sketch — refine in planning)
diff_sketch: |
  New check (~"check_repro_artifacts_clean"): parse `eval_results/issue_<N>/...` paths/globs from the
  **Repro:** footer; run `git status --porcelain -- <each subdir>`; any untracked/modified entry ->
  finding naming the files (severity per planner: FAIL at awaiting_promotion, else WARN).
## Scope / surfaces
- Primary target: `scripts/verify_task_body.py` (+ tests); keep consistent with check 29's conventions.
## Constraints / invariants
- Must tolerate legitimately-gitignored artifact classes (npz convention) — count only convention-committed types; workflow-surface only; recursion guard applies.
## Provenance
- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 81d20e797f79
Verbatim surfaced prose: "No gate compares a body's per-cell artifact claims against working-tree git status; a check FAILing when untracked/modified result files exist under a **Repro:**-named eval_results/issue_<N>/ subdir would have caught this blocker mechanically and generalizes to every awaiting_promotion task."

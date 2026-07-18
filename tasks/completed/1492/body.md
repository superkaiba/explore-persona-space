---
title: 'workflow-fix: gotchas.md bullet — SAE reference-eval token-pool semantics
  (remove_bos + outlier filter)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6f3da42ad100
created_at: '2026-07-18T05:44:51Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1482 (sae_reference_eval_token_pool_mismatch)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson (`gotcha_candidate: yes`) raised on task #1482 (emitting agent: experiment-implementer).

## Goal

Add a `.claude/rules/gotchas.md` bullet: SAE fitness/eval checks against published FVE/L0 must reproduce the reference eval's TOKEN-POOL semantics (dictionary_learning `remove_bos` 8-position strip + >10x-median-norm row filter + var-based FVE), not just the encode/decode math.

## Workflow gap

- **Bug observed:** #1482 attempt 1 burned a 4xA100 GCP cycle on a Gate-B HALT (FVE -7,704 vs published 0.806; L0 286 vs 60) caused by pooling ALL token positions into the SAE fitness read; the encoder was verbatim reference-identical (andyrdt/dictionary_learning@andyrdt/qwen batch_top_k.py) — Qwen massive-activation tokens explode L0 through the fixed scalar threshold (~42k features on a 30x-norm row) and drive FVE to -10^3, mimicking a loader/scale bug.
- **Why it is a workflow gap:** gotchas.md carries the codebase-trap catalog agents load when writing eval/analysis code; it has zero coverage of SAE reference-eval token-pool semantics, so the next SAE-consuming experiment re-derives this from a burned pod cycle.
- **Confidence (emitter):** high (root_cause_confirmed: yes; fix verified locally FVE 0.7387 + review PASS @ 2ca8cd7514)
- verified-at-filing: `grep -n 'remove_bos' .claude/rules/gotchas.md; grep -in 'token.pool|BOS strip' .claude/rules/gotchas.md; grep -cin 'sparse autoencoder|SAE' .claude/rules/gotchas.md` → 0 hits / 0 hits / 0 mentions in the named target (2026-07-18; absence-of-coverage claim — 0-hit in-target IS the evidence; not a text-matching-guard absence, no executable predicate exists for a prose catalog)

## Proposed change (candidate diff sketch — refine in planning)

+ In .claude/rules/gotchas.md, new bullet under the eval/analysis traps section:
+ **SAE reference-eval token-pool mismatch.** Comparing SAE reconstruction FVE/L0 on our activations
+ against a suite's published eval requires reproducing the reference's TOKEN-POOL semantics, not just
+ its encode math: dictionary_learning suites (e.g. andyrdt Qwen2.5-7B) train AND eval under remove_bos
+ (first 8 positions dropped) + `norms <= 10x median` row filtering, with var-based FVE; raw activations
+ are the correct input (normalize_activations is weight-folded at save). Pooling ALL positions explodes
+ L0 through the fixed scalar threshold on Qwen massive-activation tokens and reads FVE ~ -10^3,
+ mimicking a loader bug (#1482: a 4xA100 cycle burned on exactly this). Probe a deliberately-poisoned
+ pool to confirm the outlier filter engages.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'remove_bos' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; gotchas row caps respected (ratchet budget if over headroom).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 6f3da42ad100

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: P2-pilot SAE encode
lesson: The andyrdt Qwen2.5-7B SAEs' encode/decode consume RAW residual activations (normalize_activations is weight-folded at save), but their training AND published eval exclude the first 8 token positions and >10x-median-norm rows (dictionary_learning remove_bos) — pooling ALL positions into an SAE fitness read explodes L0 through the fixed threshold (~42k features on a massive-norm row) and drives FVE to -10^3. Reproduce the reference's TOKEN-POOL semantics (BOS strip + outlier filter + var-based FVE), not just its encode math, before comparing against published FVE/L0.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->

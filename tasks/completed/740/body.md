---
title: 'workflow-fix: vectorized_mlp_skill.py helper referenced but missing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6e8565604a68
created_at: '2026-06-29T23:32:55Z'
has_clean_result: false
origin_prompt: search for vectorized code — vectorized_mlp_skill.py cited by CLAUDE.md
  + vectorize-many-cell-fits.md does not exist in the repo
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed gap (confirmed empirically while locating the vectorized fit code for a #658 re-analysis).

## Goal

Fix the phantom canonical-helper reference: `.claude/rules/vectorize-many-cell-fits.md` and `CLAUDE.md` both cite `src/explore_persona_space/analysis/vectorized_mlp_skill.py` as the canonical vectorized many-cell-fit helper, but that file does not exist in the repo. Either create the helper or repoint the references to the actual vectorized code.

## Workflow gap

- **Bug observed:** `git ls-files | grep -i vectoriz` returns ONLY `.claude/rules/vectorize-many-cell-fits.md` — there is NO `vectorized_mlp_skill.py` anywhere in the tree. Yet the rule (lines ~62, ~83) and the CLAUDE.md "vectorize-many-cell-fits" / compute-character carve-out bullet both name `src/explore_persona_space/analysis/vectorized_mlp_skill.py` as "the canonical helper." An agent instructed to "use the canonical helper" cannot find it.
- **Why it is a workflow gap:** the rule + CLAUDE.md are workflow surface; they point at a non-existent file, so the guidance is unfollowable as written. The actual vectorized many-cell MLP fit currently lives inline in per-issue scripts (e.g. `_fit_mlp_ensemble_loco`, the `torch.vmap` ensemble, in `scripts/issue658_fit_predictors.py`), not in a shared helper.
- **Confidence (emitter):** high (file absence is verified by `git ls-files`)

## Proposed change (candidate diff sketch — refine in planning)

Planner picks ONE:
- (a) CREATE `src/explore_persona_space/analysis/vectorized_mlp_skill.py` as the reusable helper, factoring the vmap-ensemble LOCO MLP out of `scripts/issue658_fit_predictors.py::_fit_mlp_ensemble_loco` (the existing, exactness-tested implementation), and have the rule point at it; OR
- (b) REPOINT the rule + CLAUDE.md references to the actual in-repo vectorized code (`scripts/issue658_fit_predictors.py::_fit_mlp_ensemble_loco`) and drop the "canonical helper at <path>" framing until a shared helper is genuinely extracted.

Option (a) is the better long-term fix (the rule's intent is a REUSABLE helper); (b) is the honest stopgap. Either removes the dangling reference.

## Scope / surfaces

- `.claude/rules/vectorize-many-cell-fits.md`
- `CLAUDE.md` (the vectorize-many-cell-fits / compute-character carve-out bullet under "CPU-only phases don't hold GPU pods")
- (if option a) `src/explore_persona_space/analysis/vectorized_mlp_skill.py` (NEW) — note this path is OUTSIDE the normal workflow surface, but creating the file the rule mandates is in-scope for closing the gap; the planner decides (a) vs (b).
- Grep first: `grep -rln 'vectorized_mlp_skill' .claude/ CLAUDE.md` to catch every reference.

## Constraints / invariants

- Keep the rule's `paths:` frontmatter and CLAUDE.md wording consistent with whichever option is chosen.
- If option (a): the extracted helper must preserve `_fit_mlp_ensemble_loco`'s exactness contract (the `_assert_mlp_exactness` oracle) — do not silently change fit numerics.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/vectorize-many-cell-fits.md
- fingerprint: 6e8565604a68

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/vectorize-many-cell-fits.md, CLAUDE.md
bug_observed: git ls-files shows no vectorized_mlp_skill.py anywhere in the repo, yet .claude/rules/vectorize-many-cell-fits.md and CLAUDE.md both name it as the canonical helper, so an agent told to use the canonical helper cannot find it.
why_workflow_gap: the rule + CLAUDE.md (both workflow surface) point at a non-existent file; the actual vectorized many-cell MLP fit lives inline as _fit_mlp_ensemble_loco in scripts/issue658_fit_predictors.py.
proposed_change: Either create src/explore_persona_space/analysis/vectorized_mlp_skill.py by factoring out _fit_mlp_ensemble_loco, or repoint the rule + CLAUDE.md references to the actual in-repo vectorized code; remove the dangling reference either way.
diff_sketch: |
  - `src/explore_persona_space/analysis/vectorized_mlp_skill.py` — the reusable
  + (option b) the canonical vectorized many-cell fit lives in
  +   `scripts/issue658_fit_predictors.py::_fit_mlp_ensemble_loco` (torch.vmap
  +   ensemble, exactness-tested); extract to a shared helper when reused again.
confidence: high
related_task: #658
<!-- /workflow-fix-candidate -->

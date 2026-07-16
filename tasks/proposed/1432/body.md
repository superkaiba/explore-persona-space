---
title: 'workflow-fix: gotchas.md — cuda eigh non-convergence on near-singular Grams
  (CPU fallback)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5f9438da5701
created_at: '2026-07-16T18:04:24Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1335 r10 crash-fix (cuSOLVER
  eigh LinAlgError at the matched-n inner-fold Gram); routed per workflow-fix-on-bug'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes`
failure-lesson raised on task #1335 (emitting agent: issue-session orchestrator,
r10 crash-fix round).

## Goal

Add a gotchas.md entry: cuda `torch.linalg.eigh` (cuSOLVER syevd) raises
`LinAlgError` on near-singular / repeated-eigenvalue Grams that CPU LAPACK
handles — wrap Gram eigh sites in a CPU-fallback helper.

## Workflow gap

- **Bug observed:** `issue825_fit_cells.py _prep_inner_lambda` cuda eigh
  failed to converge on a matched-n subsampled inner-fold Gram
  (`torch._C._LinAlgError: linalg.eigh: The algorithm failed to converge
  because the input matrix is ill-conditioned or has too many repeated
  eigenvalues`), killing the #1335 instruct matched lane (attempt 8,
  2026-07-16T17:55Z); the SAME code had run thousands of larger full-lane
  eigh calls cleanly — the trigger is SMALL subsampled group-split blocks
  with near-duplicate rows. Fixed by `_eigh_robust` (d1922d2068).
- **Why it is a workflow gap:** `.claude/rules/gotchas.md` has no entry for
  this cuSOLVER quirk, and 10+ repo scripts call `linalg.eigh` on
  data-derived Grams (issue823/779/667/493/722/532/825 pipelines) with the
  same latent exposure — each will re-hit and re-diagnose it independently
  on the next subsampled/group-split fit.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rn 'eigh' .claude/rules/gotchas.md` → 0 hits
  (2026-07-16; absence-of-guard claim — the 0-hit in-target result IS the
  evidence the gotcha is undocumented); exposure survey
  `grep -rln 'linalg.eigh' scripts/ src/ --include='*.py'` → 10+ files.

## Proposed change (candidate diff sketch — refine in planning)

```
+ ## cuda torch.linalg.eigh fails to converge on near-singular Grams
+
+ cuSOLVER's syevd raises torch.linalg.LinAlgError ("failed to converge ...
+ ill-conditioned or has too many repeated eigenvalues") on near-singular
+ Grams that CPU LAPACK decomposes fine. Trigger regime: SMALL subsampled /
+ group-split fold Grams with near-duplicate rows (a full-size run can pass
+ thousands of eigh calls, then die at matched-n subsampling — #1335 r10).
+ Wrap Gram eigh sites in a CPU-fallback helper (try cuda eigh, except
+ torch.linalg.LinAlgError -> eigh(G.cpu()) moved back to device) — exact
+ backend swap, fp-roundoff agreement; do NOT jitter the Gram (changes the
+ numbers). Canonical: scripts/issue825_fit_cells.py::_eigh_robust.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'linalg.eigh' .claude/ CLAUDE.md scripts/ src/`); the entry
  should name the canonical helper (`issue825_fit_cells.py::_eigh_robust`,
  landing on main with the #1335 merge) rather than prescribe per-site
  hand-rolled try/excepts. A complementary agent-memory
  `feedback_cusolver_eigh_nonconvergence_cpu_fallback.md` was committed at
  12bc4eee67.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 5f9438da5701

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: issue825_fit_cells.py _prep_inner_lambda cuda eigh failed to converge on a matched-n subsampled inner-fold Gram, killing the #1335 instruct matched lane; 10+ repo scripts call linalg.eigh with the same exposure
why_workflow_gap: gotchas.md has no entry for cuSOLVER syevd non-convergence on near-singular Grams, a latent trap in every subsampled/group-split spectral fit in the repo
proposed_change: Add a gotchas.md entry: cuda torch.linalg.eigh (cuSOLVER syevd) raises LinAlgError on near-singular / repeated-eigenvalue Grams that CPU LAPACK handles — wrap Gram eigh sites in a CPU-fallback helper
diff_sketch: |
  + ## cuda torch.linalg.eigh fails to converge on near-singular Grams
  + Wrap Gram eigh sites in a CPU-fallback helper (try cuda eigh, except
  + torch.linalg.LinAlgError -> eigh(G.cpu()) moved back) — exact backend
  + swap; do NOT jitter. Canonical: issue825_fit_cells.py::_eigh_robust.
confidence: high
related_task: #1335
<!-- /workflow-fix-candidate -->

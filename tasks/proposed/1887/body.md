---
title: 'harden ridge lambda-selection: inner-group-CV default + GCV dof-cap + n<d
  degeneracy tripwire + re-read audit of affected published cells'
kind: infra
tags: []
created_at: '2026-07-30T18:07:13Z'
has_clean_result: false
origin_prompt: 'Can you look deeper into this: [the GCV/reduced-basis finding] and
  make sure it doesn''t happen again. Explain the problem and also what the solution
  could be (user chat, 2026-07-30)'
workflow: v1
---
## Overview / Motivation

The story-context-info-probe round on #1345 (2026-07-30; eval_results/issue_1345/story_context_info_probe/, commits 3ffc51d581/9f0fb74d4a) proved the published #1345 story-collapse R2 values are an estimator artifact: story v_C -> story v_A reads -0.547 under the published rig (ambient d=3,584 pure-GCV ridge, n_train < d) but +0.262 in a train-fold reduced PCA basis and +0.44 in the AMBIENT basis at forced lambda=1e3 (forced_lambda_probe.json). Mechanism: at n_train < d the ridge smoother can (near-)interpolate — RSS -> 0 while (n - dof)^2 -> 0 — so the GCV objective RSS/(n-dof)^2 degenerates and its minimum lands at/near the lambda-grid's lower edge; the resulting near-interpolating map fails held-out catastrophically (negative R2) while the information is demonstrably present (raw retrieval ~300x chance, CCA 0.84-0.90). n_train > d cells (e.g. #825 full-corpus, n=4,724 > 3,584) are structurally immune, which is why the instrument validated cleanly and the failure surfaced only in matched-row / small-cell refits where n silently dropped below d.

**The fix already exists and is stranded off-default** (#1417 `registered-selector-refit` round): `issue825_map_alignment.py` lines ~97-124 + `issue825_fit_cells.py` carry `LAMBDA_SELECTION = "inner-group-cv"` machinery and `GCV_DOF_CAP` (interpolating-lambda skip), defaults `"gcv"` / `None` "byte-identical committed behavior". Every post-#1417 fit line (#1345 rounds incl. the story headline cells, #1310 per-persona fits, #1639 per-cell refits) kept pure GCV. This is the CLAUDE.md "built-but-stranded fixes" class.

## Goal

Make pure-GCV lambda selection at n_train < d impossible to run silently again, and re-read the affected published cells under the corrected instrument.

## Deliverables

1. **Flip the defaults in both fit cores** (`scripts/issue825_map_alignment.py`, `scripts/issue825_fit_cells.py`): `LAMBDA_SELECTION="inner-group-cv"` default; GCV usable only WITH a dof cap (default 0.9) and only via explicit opt-in; per-fit selected-lambda logging ON by default into cell JSONs (the SELECTOR_LOG hook exists, default None). Callers relying on byte-identical committed behavior pass an explicit `legacy_gcv=True`-style opt-in.
2. **Degeneracy tripwire in the shared fit path:** WARN-tag the cell JSON (`estimator_degenerate_suspect: true`) and print loud when (a) n_train < d AND the selected lambda sits within one grid step of the grid's lower edge, or (b) held-out R2 < 0 while the same cell's kNN retrieval exceeds ~20x chance (the dissociation signature). Plus: auto-compute the reduced-basis companion (k=min(1024, floor(n_train/2)), train-fold basis) whenever n_train < d — promote the #1701 dispatch-note duty to a code default.
3. **Tests** pinning the new defaults + the tripwire (synthetic n<d fixture where pure GCV demonstrably picks the grid edge and inner-CV does not).
4. **0-GPU re-read audit of affected published cells** from the pinned HF stores (reduced-basis + lambda-sweep + inner-CV reads beside the committed ambient-GCV values): #1345 matched-row story/chat/plain-text cells (incl. slot-ablation + CJK-refit), #1310 per-persona cells (n approx 300-3,600 < d), #1639 per-cell refits. #1335 used inner-group-cv (safe); #1689 uses grid+inner-3-fold CV (safe from THIS artifact). Output: one corrections table per issue naming which published numbers move and which verdicts (framing-effect CIs, reach-bar rung reads, recovery fractions vs matched ceilings) survive; fold into the affected bodies per the refuted-body duty.
5. **Rule addendum** (workflow surface, small): extend the estimator-validity duty (CLAUDE.md inline-round duties, #1701 clause) with the GCV-specific ban: pure-GCV lambda selection at n_train < d is refused; selected-lambda diagnostics are reported alongside every ridge read.

## Evidence / verified-at-filing

- `grep -n 'LAMBDA_SELECTION\|GCV_DOF_CAP' scripts/issue825_map_alignment.py` -> defaults `"gcv"` / `None` at lines 106-107 (2026-07-30); same pattern in `scripts/issue825_fit_cells.py` (GCV_DOF_CAP=None line ~91).
- #1417 body Methodology table row "lambda selection": run 1 pure GCV ("the plan's small-cell fallback unwired"), refit rounds inner-group-cv + dof-capped fallback — the incident precedent.
- Probe artifacts: `eval_results/issue_1345/story_context_info_probe/{summary.json,forced_lambda_probe.json}`.
- In-flight mitigations already applied (not this task's scope): the #1345 story-boundary-ablation round's fits are getting reduced-basis + forced-lambda companions with the reduced read primary; the #1689 user-slot-recapture round's fits spec reduced companions (addendum E); #1689 round 4 (wellposed-shared-readout) addresses the sibling ambient-shrinkage class.

## Constraints

- Do NOT change published eval JSONs; the audit writes NEW companion files + a corrections table.
- The defaults flip must not silently alter any RUNNING round's in-flight fits: land after coordinating with the live #1345/#1689 rounds (their fits already carry the corrected reads explicitly).
- est_gpu_hours: 0 (CPU refits on pinned stores).

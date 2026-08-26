---
title: 'artifact-reuse: prove a reused GATE''s committed anchor is reproducible under
  the shared core''s CURRENT defaults (#1887 flip silently changed #2546''s G-E meaning)'
kind: infra
tags: []
created_at: '2026-08-26T11:31:52Z'
has_clean_result: false
origin_prompt: 'Surfaced by task #2546 arm-1 G-E gate halt 2026-08-26: the plan named
  issue1336_fit_cells.py --g0 as the instrument for the committed 0.6731 anchor, but
  the #1887 shared-core defaults flip (lambda_selection -> inner-group-cv) means that
  entry point no longer reproduces the anchor; the gate FAILed while the fit core
  was in fact byte-exact against #1336''s own committed value.'
workflow: v1
---
---
kind: infra
---

# Artifact-reuse checklist: a reused GATE's committed anchor must be proven reproducible under the shared core's CURRENT defaults

## Goal

Close a concrete gap in `.claude/rules/artifact-reuse.md`: nothing in checks (a)-(m) requires proving that a REUSED GATE's committed reference value is still reproducible under the shared fit core's **current** defaults. When a shared core changes a default, every downstream gate pinned to a pre-change committed number silently changes meaning and then FAILs for a reason that looks like a reuse/device failure but is not.

## The incident that produced this (task #2546, arm 1, 2026-08-26)

Plan v4 §7 adopted #1336's G-E fit-core reuse gate verbatim: run `issue1336_fit_cells.py --g0`, refit the pinned #825 Qwen S1 cell @ revision `deb7a4523b5233393e4fbd2497622527b3622d35`, PASS iff layer-19 held-out R² is within ±0.01 of the committed 0.6731. It doubled as the artifact-reuse check-(m) device-domain exercise.

It FAILed on the primary arm: `[g0] layer-19 R2=0.6935 vs committed 0.6731 (tol 0.01) -> FAIL`, rc=3, halting the pipeline before the fits phase.

**Nothing was broken.** The measured value `0.6935026836671432` is byte-identical, to all 16 significant figures, to #1336's own committed `eval_results/issue_1336/gates_v2/g0v2.json` → `leg_b_gram_vs_primal.r2_primal` and `leg_c_v2_anchor.s_qwen_v2`, computed on different hardware four weeks earlier. The reused core reproduces bit-for-bit; the gate compared that number against an anchor belonging to a different estimator.

Mechanism: `scripts/issue825_fit_cells.py::heldout_r2_sweep` documents the **#1887 defaults flip** — `lambda_selection` now defaults to `"inner-group-cv"`, and its own docstring says "Committed pre-#1887 behavior needs the explicit legacy pins". `issue1336_fit_cells.py::run_g0` passes NO `lambda_selection` and NO `lambdas`, so it silently follows whatever the shared core currently defaults to. The 0.6731 anchor is reachable ONLY under the legacy pins (`lambda_selection="gcv"`, `lambdas=logspace(-2,4,13)`, `GCV_DOF_CAP=None`, `LEGACY_UNGUARDED_GCV=True`, `FORCE_GRAM=True`).

Dated proof that the instrument went stale rather than being wrong from birth:
- `eval_results/issue_1336/gates/g0_gate.json` — 2026-07-16T02:54:20Z, commit f1edfece: r2 0.6730940896676356, abs_dev 5.91e-06, **pass true**. Ran while the core still defaulted to legacy GCV.
- `eval_results/issue_1336/gates_v2/g0v2.json` — 2026-08-02T22:05:38Z: records BOTH numbers with pins made explicit and adds a legacy-pinned leg (a) specifically to keep the anchor comparison alive. Its own docstring: "run_g0 itself is left untouched — v1 gate unchanged."

So #1336 recognized the problem and routed around it for itself, but the v1 `--g0` entry point it left in place is the one a later plan naturally reaches for — and #2546's plan did.

Three-way separation measured on the identical cell (#2546, same bundle / layer 19 / seed 0 / n 5000):

| configuration | R² | abs_dev vs 0.6731 |
|---|---|---|
| legacy GCV, `logspace(-2,4,13)`, dof-cap off, unguarded, gram forced | 0.6730940896676356 | 5.9e-06 |
| `inner-group-cv`, 23-pt grid, 2 inner folds | 0.6935026836671432 | 0.0204 |
| `inner-group-cv`, 23-pt grid, 4 inner folds | 0.6957042061410352 | 0.0226 |

Neither fold count reaches the anchor: the anchor depends on the SELECTOR and its grid, not the fold count.

## Cost of the gap

A pre-registered kill criterion fired on the primary arm of a 111-GPU-hour experiment, halting it, and the diagnosis required reading four source files plus two committed artifacts across four git refs to establish that nothing was wrong. A future reader of the `epm:failure` marker alone ("layer-19 held-out R2 outside +/-0.01 of 0.6731") would reasonably conclude the reused fit core failed validation — the exact opposite of the truth.

## Proposed change

Add to `.claude/rules/artifact-reuse.md` a check in the (a)-(m) family — a REUSED-GATE ANCHOR check — requiring that when a plan adopts an existing gate whose PASS condition is a committed numeric reference:

1. The plan states which estimator configuration produced that reference (selector, grid, and every pinned knob), citing where it is recorded.
2. The gate either (a) PINS that configuration explicitly rather than inheriting shared-core defaults, or (b) carries evidence that the reference is still reproducible under current defaults — a dated artifact, not an assumption.
3. A reference whose producing configuration cannot be established is not usable as a gate bar; the gate is re-derived or the anchor re-measured under the current regime.

Recommended strengthening (what #2546 is implementing for itself): a two-leg gate — leg 1 pins the legacy regime and checks the historical anchor, leg 2 checks the CURRENT recipe against its own known value. Leg 1 certifies the anchor, leg 2 certifies recipe + device identity. Both enforced. #2546's run already reproduced the current-recipe value byte-exactly, so the second leg is free and strictly more informative than the single-leg form.

Consider also whether `heldout_r2_sweep`-family entry points used as GATES should refuse to run without an explicit `lambda_selection`, so silent default-inheritance in a gate becomes impossible rather than merely discouraged. That is a shared-core change and needs its own review; it is raised here as a question, not a prescription. **Do not revert or alter the #1887 flip itself** — it is registered behavior other issues depend on.

## Scope

Rule-surface change plus, if warranted after review, a lint or gate-side guard. Not an experiment; no GPU. Do NOT edit `scripts/issue825_fit_cells.py` or `scripts/issue1336_fit_cells.py` behavior as part of this task without a separate review — they are shared reused cores.

## Provenance

Surfaced by task #2546 (arm 1, G-E gate halt, 2026-08-26). Full diagnosis with all evidence: #2546 `epm:progress` markers v88 and v89. Related concern rows on #2546: `caphit-per-cell-rates-must-reach-the-digest`, `gb-repetition-gate-slice-is-least-affected-corpus`.

---
title: 'workflow-fix: noise-structure symmetry — shared-baseline cos(X−B,Y−B) DVs
  vs nulls'
kind: infra
tags:
- wf-fix
- wf-fix-fp:87daf963d358
created_at: '2026-07-17T04:18:22Z'
has_clean_result: false
origin_prompt: 'interp-critic prose follow-up on #1415: shared-baseline difference-vector
  cosine DVs need disjoint baseline draws or a baseline-noise-bearing null (headline-affecting
  inflation caught at interpretation-critique)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised by the interpretation-critic on task #1415 (2026-07-17, interp-critique v1).

## Goal

Add a rule entry (primary: `.claude/rules/selection-symmetric-nulls.md`; the planner may instead/also place a `.claude/rules/gotchas.md` bullet) covering shared-baseline difference-vector cosine DVs: `cos(X − B, Y − B)` where both legs subtract the SAME sampled baseline B carries shared-baseline noise that a noise-free null (e.g. random-direction projections) does not bear — the DV must use DISJOINT baseline draw halves for the two legs, or the null must be constructed with the same shared-B structure.

## Workflow gap

- **Bug observed:** #1415's H1 answer-shift alignment cosine subtracted one shared 10-draw V_a(c) baseline mean from BOTH the realized shift and the target shift, while the norm-matched random-Δ null had no shared term by construction; the interp-critic's disjoint-baseline recount dropped canonical alignments (prefix 0.271→0.178, context 0.362→0.272), sent one pair fully artifactual (0.23→−0.08), and pulled 6/28 prefix pairs below the null — a headline-affecting inflation that survived the planner, the critic ensemble, the implementer, and code review, caught only at interpretation-critique.
- **Why it is a workflow gap:** `.claude/rules/selection-symmetric-nulls.md` covers selection symmetry (inherit the max/argmax per draw) but has no clause on NOISE-STRUCTURE symmetry between an observed difference-vector DV and its null; nothing in the rules names the shared-baseline cosine trap, so future geometry DVs (this project's bread and butter) can re-hit it.
- **Confidence (emitter):** high (the recount is executed and verified on #1415's persisted per-draw tensors)
- verified-at-filing: `grep -rn -iE "shared.baseline|difference.vector|cos\(X" .claude/rules/selection-symmetric-nulls.md .claude/rules/gotchas.md` → 0 hits in both targets (absence-of-guard claim — the 0-hit in-target result IS the evidence) (2026-07-17)

## Proposed change (candidate diff sketch — refine in planning)

+ selection-symmetric-nulls.md, new section "Noise-structure symmetry (shared-baseline difference vectors)":
+   A cosine/projection DV of the form cos(X − B, Y − B) (or any statistic whose observed and
+   reference legs share one SAMPLED baseline B) inherits positive bias from the shared noise in B;
+   a null lacking that shared term (random directions, shuffled pairs re-centered independently)
+   under-covers. EITHER split B's draws into disjoint halves feeding the two legs (unbiased, the
+   #1415 recount recipe), OR build the null to carry the identical shared-B structure per draw.
+   Incident #1415: 28/28-clear-the-null headline -> 6/28 below null after disjoint recount.

## Scope / surfaces

- Primary target: `.claude/rules/selection-symmetric-nulls.md`
- Sibling to consider: `.claude/rules/gotchas.md` (analysis-code trap register); `planner.md` §6 / statistics-critic lens pointer if the planner surface names null construction.
- Grep before editing: `grep -rn -i "null" .claude/rules/selection-symmetric-nulls.md` + the lens-coverage map if a lens pointer is added.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` default run passes (incl. --check-lessons-index if a new rule file is created rather than extended).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/selection-symmetric-nulls.md
- fingerprint: 87daf963d358

Verbatim surfaced prose (interp-critique v1 return, #1415): "One prose follow-up surfaced for the orchestrator: a candidate `.claude/rules/` lesson on shared-baseline difference-vector cosines (cos(X−B, Y−B) DVs need disjoint draws or a baseline-noise-bearing null)."

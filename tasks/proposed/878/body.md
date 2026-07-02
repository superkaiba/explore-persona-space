---
title: 'workflow-fix: design-aligned split-half clause for crossed designs (llm-judging.md)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:696b59c95925
created_at: '2026-07-02T17:46:20Z'
has_clean_result: false
origin_prompt: 'interpretation-critic prose follow-up on #763 reanchor round: llm-judging.md
  section F needs a design-aligned split-half clause — independent per-condition item
  halves on crossed designs bias r_hh negative and clip real ceilings to 0 (#763 incident)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #763 (emitting agent: interpretation-critic, reanchor-round critique v3).

## Goal

Add a "design-aligned split-half" clause to `.claude/rules/llm-judging.md` § F (validation as a measurement instrument) requiring item-ALIGNED half-splits across conditions when computing split-half reliability over a CROSSED design.

## Workflow gap

- **Bug observed:** per-condition-independent probe-half splits on a crossed context × probe design leak probe main effects into split noise (probe-mean sd ≈ 27/100 vs context signal ≈ 3/100), biasing split-half r systematically negative; Spearman–Brown clipped a real reliability ceiling (0.59) to 0.00 in #763 (`scripts/issue_763_reliability.py` / `reliability_split_half_over_probes`).
- **Why it is a workflow gap:** no rule clause mandates item-aligned half-splits for crossed designs, and every future √r_yy in this program uses the same crossed context×probe design — the artifact will recur silently, censoring real signal.
- **Confidence (emitter):** high (the critic independently reproduced the mechanism: v1 split-half −0.41 / v2 −0.23, ~100%/99% of 200 splits negative; probe-aligned split gives v2 ceiling ≈0.59).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/rules/llm-judging.md § F, new rule (after rule 15):
+ - **Design-aligned split-half for crossed designs.** When the reliability
+   ceiling (split-half + Spearman–Brown) is computed over a CROSSED design
+   (the same item/probe set scored under every condition), the half-split
+   MUST be item-ALIGNED across conditions (both halves contain the same
+   probes for every condition). Splitting each condition's items
+   independently leaks item main effects into the split noise, biases the
+   split-half correlation systematically negative, and the Spearman–Brown
+   clip reads a real ceiling as 0 (incident #763: probe sd ≈27/100 vs
+   context signal ≈3/100 clipped a 0.59 ceiling to 0.00).
```

## Scope / surfaces

- Primary target: `.claude/rules/llm-judging.md`
- Grep the workflow surface for sibling reliability/split-half prose before editing (`grep -rln 'split-half' .claude/ CLAUDE.md scripts/`) and update every in-scope hit; list them in the plan. (The analysis-code fix in `issue_763_reliability.py` is experiment code — OUT of scope here; it is already captured as #763's free-analysis follow-up.)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/llm-judging.md
- fingerprint: 696b59c95925

Surfaced prose (verbatim, from the interpretation-critic's return on #763): "`.claude/rules/llm-judging.md` § F (or a sibling stats rule) should gain a 'design-aligned split-half' clause: when computing split-half reliability over a CROSSED design (same items/probes under every condition), the half-split must be item-ALIGNED across conditions, otherwise item main effects (here probe-mean sd ≈27/100 vs context signal ≈3/100) bias r_hh systematically negative and Spearman–Brown clips a real ceiling to 0 — exactly the #763 `issue_763_reliability.py` artifact, and every future √r_yy in this program uses the same crossed context×probe design, so it will recur."

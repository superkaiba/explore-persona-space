---
title: 'Absolute per-cell trainability floor: a yield shortfall must DROP the cell,
  not shrink it to 1 row'
kind: infra
tags: []
created_at: '2026-08-12T10:32:05Z'
has_clean_result: false
parent_id: 2221
origin_prompt: 'surfaced by #2221: evil training mixes realized 1 row per cell and
  all 24 cells still fine-tuned, judged, and correlated with no gate firing'
workflow: v1
---
# Add an absolute per-cell trainability floor to the yield-shortfall contract

## Goal

Close the gap that let a fine-tune cell train on **1 row** and flow all the way to
monitor correlations with no gate firing: the yield-shortfall contract specifies
"shrink + report" with no ABSOLUTE minimum below which a cell is DROPPED and the
experiment's denominator revised.

## The incident (#2221)

#2221 built 24 real-data-twin training mixes by judge-banding found/organic responses,
then equalizing down within family to floor-N. Realized per-cell rows:

| family | rows/cell |
|---|---|
| **evil** | **1** |
| **mistake_opinions** | **13** |
| sycophancy | 175 |
| mistake_medical | 233 |
| mistake_gsm8k | 520 |
| mistake_math | 760 |
| hallucination | 1,533 |
| insecure_code | 3,357 |

Plan §4 targeted "~a few k" rows per cell. Evil got 1. All 24 cells were fine-tuned
(rs-LoRA r32, lr 1e-5, 1 epoch), captured, judged with a 6-draw Sonnet graded rubric,
and carried into the monitor-arm correlations and a checkpoint-detection AUC.

Downstream cost: evil's DV floored at 0.00-1.21 (base 0.00) and mistake_opinions showed
no install, so 6 of 24 cells were effectively untrained; the only trait-acquiring cells
were the 2 largest families, making "which cells acquired the trait" substantially a
proxy for "which family had enough rows" — a confound on the headline comparison, plus
wasted judge spend on 6 non-conditions.

## Why nothing fired

Plan §7's gate row read `Per-cell realized yield >= floor-N (else shrink+report)`,
explicitly "graceful shrink, never abort". The related kill criterion required **ZERO**
band-II rows for **>= 2 of 3** chat-trait families; evil produced 1 row (not zero) and
the other two chat families were healthy, so no criterion was met. The pipeline
followed the plan exactly.

`.claude/rules/on-policy-completions.md` and `.claude/rules/data-realism.md` both say a
below-floor shortfall is REPORTED and the source DROPPED — never silently carried. The
plan's "shrink to whatever survives" wording is compatible with the letter of
"shrink+report" while violating that intent, and the critic ensemble did not flag it.

## Proposed change (implementer to scope)

1. **An absolute trainability floor, separate from the equalize-down floor-N.** Below
   it a cell is DROPPED, not shrunk: the cell does not train, the denominator is revised
   everywhere it appears, and the drop is named in the clean-result `## Takeaways` per
   the After-Every-Experiment planned-vs-actual rule. Ground the default in literature /
   prior issues rather than inventing a constant (the LoRA-install evidence in
   `.claude/rules/marker-training-recipe.md` is the nearest in-repo anchor for "how many
   rows install anything at a given lr/epoch"); a plan may override it explicitly with
   a stated argument, which is exactly the visibility that was missing here.
2. **A pre-P4 hard assert in the mix-build/fine-tune dispatch path** — a cell whose
   realized row count is below the floor fails loud BEFORE any GPU or judge spend,
   rather than producing a silent non-condition. This is the fail-fast placement: the
   gate belongs where the rows are counted, not in post-hoc interpretation.
3. **A planner/critic surface rule** so a plan's yield row must state the absolute
   floor and the DROP disposition, not only the shrink path — the review-side arm that
   would have caught this plan's wording.

Scope note for the implementer: prefer extending the existing yield/band-count
machinery over adding a parallel gate, and check whether a shared helper already
counts realized per-cell rows before writing a new one (reuse-discovery rule).

## Provenance

Surfaced by the #2221 orchestrator while composing that issue's clean-result: the
1-row evil mixes were found by counting the realized mixes on disk and confirming them
byte-identical to the HF copies.

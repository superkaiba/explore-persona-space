---
title: 'api_throughput_guidelines: disambiguate the two sync/batch crossover figures
  (200k balanced vs 20k dispatcher constant) + state precedence vs CLAUDE.md'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-19T21:36:14Z'
has_clean_result: false
origin_prompt: 'surfaced by the #2329 q35_ladder_decay critique round 1: Claude cited
  ~200k and Codex cited 20k from the same doc in support of opposite verdicts on the
  same plan, costing a binding reconciler spawn; the reconciler found both figures
  are really there for different objects and the doc does not distinguish them'
workflow: v1
---
# api_throughput_guidelines: disambiguate the TWO sync/batch crossover figures that reviewers cite interchangeably

## Provenance

workflow_fix_target: docs/api_throughput_guidelines.md

Surfaced by the #2329 `q35_ladder_decay` post-approval critique panel (round 1),
where it caused a full Claude-vs-Codex lens disagreement that consumed a binding
reconciler spawn to resolve.

## The bug

`docs/api_throughput_guidelines.md` contains TWO different sync/batch crossover
numbers describing two different objects:

- **~200k** — the Sonnet-4.5 balanced crossover (around line 75).
- **20k** — the recommended dispatcher constant `SYNC_BATCH_CROSSOVER_N`
  (around line 132).

Nothing in the document flags that these are different quantities. In the #2329
round-1 panel, the Claude efficiency critic cited ~200k and the Codex efficiency
twin cited 20k, each in support of an OPPOSITE verdict on the same plan
(PASS vs REVISE) — Codex additionally proposed a mechanical check that would
"reject forced Batch when passes exceed 3 or N < 20,000", which would have
contradicted CLAUDE.md's standing Batch mandate had it been implemented.

The reconciler's finding was that neither reviewer misread the document: both
figures are really there, for different objects, and the document does not
distinguish them. That is a documentation defect, not a reviewer defect — and it
cost a reconciler spawn plus two rounds of critic reasoning.

## Secondary gap in the same document

The reconciler also had to establish the document's own AUTHORITY from a
self-description buried at lines 10-12 ("RECOMMENDATIONS, not hard API limits")
in order to rule that CLAUDE.md's Batch mandate and the user's scope directive
both outrank it. That precedence is correct but should not require excavation:
the document should state its standing relative to CLAUDE.md's Critical Rules
up front.

## Proposed fix

1. Rename or qualify each figure at its site so they cannot be cited
   interchangeably — e.g. "balanced-cost crossover (Sonnet 4.5): ~200k" vs
   "dispatcher routing constant `SYNC_BATCH_CROSSOVER_N` = 20k (a conservative
   default, not the cost-balance point)" — and cross-reference each from the
   other.
2. Add a short precedence header near the top: this document is
   recommendations; CLAUDE.md Critical Rules and an explicit user directive in a
   task's scope marker both outrank it; the wedge-pass guidance is a general
   default that a calendar-tolerant zero-GPU phase may legitimately override.
3. State explicitly that the wedge-pass rule (>2-3 passes => prefer sync)
   applies to GPU-coupled or latency-coupled waves, and that a phase holding no
   compute may take Batch above that pass count — the exact case the #2329
   reconciler had to reason out from first principles.

## Acceptance criteria

1. Each crossover figure carries a distinguishing name at its definition site,
   and neither can be read as "the" crossover.
2. A precedence statement appears before the decision table.
3. The wedge-pass rule states its scope (compute-coupled vs zero-GPU
   calendar-tolerant waves).
4. No numeric recommendation CHANGES in this task — it is a disambiguation and
   precedence edit only. Any actual threshold change is a separate task with its
   own grounding.
5. Grep the repo for existing citations of either figure and update any that are
   now ambiguous.

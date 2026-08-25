---
title: 'plan-conditions check: flag ran-but-unreported registered conditions, not
  just literal slug coverage'
kind: infra
tags: []
created_at: '2026-08-24T14:45:53Z'
has_clean_result: false
parent_id: 823
origin_prompt: 'Surfaced by the /issue 823 orchestrator: the same ran-but-unreported
  shape produced clean-result blockers at rounds 8 and 9, both caught only by an LM
  critic.'
workflow: v1
---
## Goal

Upgrade the plan-conditions coverage check from literal-slug coverage to a
ran-but-unreported check: a registered condition whose results are present in
committed result JSONs but absent from the clean-result body should flag.

## Why

This exact shape produced a blocker in TWO consecutive clean-result rounds at #823,
both caught only by an LM critic, neither by any mechanical check.

Round 8: the plan-registered rung-1 capped-vs-pure-GCV sensitivity read
(`sens_estimator`) had run — `/primary/5000/sens_estimator` was present in the
committed `ladder_ext_r2.json` with 30 fold-cells and a clean null — and the body
had ZERO hits for `sens_estimator`, `capped-vs-pure`, or `pure-GCV`.

Round 9: the companion ladder's plan-mandated identity+learned-bias and kNN
retrieval reads had run for BOTH ladders (228/228 cell keys, 4/4 kNN), and the
body's only mandatory-reads table was stream-prefix-headed with the companion
ranges appearing nowhere. The binding reconciler upheld this as blocking, on the
ground that plan v17 states a REPORTING duty ("Baselines (mandatory, every fitted
map, BOTH ladders)"; the reads "reported alongside held-out R² at every
(rung × arm)"), not merely a compute duty.

The existing check flags literal slug coverage and WARNs advisorily, which both
rounds passed while the substantive omission stood.

## Scope

For each plan-registered condition slug, if its results appear in the round's
committed `eval_results/` JSONs but the slug's subject matter appears nowhere in
the body, flag it. Paraphrase coverage must not trip it: #823's 12 WARN-flagged
slugs were verified one by one as genuine paraphrase coverage, so a naive
slug-grep would produce 12 false positives on a body that is actually complete.
That is the hard part of this task and the reason it is not a one-line change.

## Acceptance

- A fixture where a registered condition's results exist in a committed JSON and
  the body never mentions the measurement FLAGS.
- A fixture where the body covers the condition in paraphrase does NOT flag.

---
name: batch-drain-drops-bare-scalar-verdicts
description: The Anthropic BATCH results drain in eval/batch_judge.py omits _normalize_scalar_score, so bare-numeric judge verdicts are silently discarded as parse_error; sync drains keep them. Route, not budget, explains anomalous graded-judge drop rates.
metadata:
  type: feedback
---

**A high `parse_error` drop rate on a BATCH-routed graded judge is a ROUTE
defect before it is a budget defect. Discriminate by re-issuing the failing
items on the SYNC path at the IDENTICAL `max_tokens` — never by raising the
budget alone.**

`eval/batch_judge.py`'s batch results drain does
`parsed = parse_judge_json(text)`; every sync drain in `judge_dispatch.py`
does `_normalize_scalar_score(parse_judge_json(text))` (the #1434
"dispatch-path parity" fix). A judge reply that is a BARE NUMBER (`85`) parses
to a Python scalar, which the dict-shaped plumbing erases to `parse_error` on
batch and keeps on sync. `_normalize_scalar_score`'s own docstring names the
trigger: the **persona-vectors rubric's "just the number" instruction
routinely wins over the JSON wrapper** — so every `load_trait_rubric`-family
graded read is exposed. The same drain also skips `_parsed_with_raw`, so the
batch path persists NO raw response text, which makes rule 23's
"diagnose from the stored raw" unexecutable for batch reads.

**Why:** #1739 item-A. Same rollouts / rubric / model / `max_tokens=1024` /
3 draws, route as the only variable — content drops batch 5.87% / **64.17%** /
2.70% vs sync 2.57% / **2.80%** / 1.83% (mhj / tom-gibbs / pair). On batch,
tom-gibbs lost 1,925/3,000 draws and 56/200 contexts entirely, non-randomly
(it censors whichever content makes the judge answer terse), on the arm whose
gate verdict sat 0.022 under the edge. Budget was ruled out: a fresh batch
re-judge at 4096 (`state.json` confirmed) left it at 63.70%. Live on `main` at
`1cb0b680b843807c371818dc6bbaa589f23948fc` (2026-08-02).

**How to apply:**
- Anomalous batch drop rate ⇒ run a ~60-item SYNC probe at the SAME budget on
  the failing items. Recovery there = route defect. Force sync via
  `judge_dispatch.decide_route`'s `threshold_base` (a large value routes sync;
  `0` forces batch) — `judge_graded` exposes `threshold_base`, not `force_sync`.
- Vary ONE thing per probe. My own first probe changed budget AND `n_draws`
  AND route at once and produced a confidently wrong "truncation-driven"
  diagnosis that the full re-run refuted — the same confound shape #1739 had
  already recorded for its item-D v1 rubric bug.
- Cross-run comparability: a sync-drained pool and a batch-drained pool are
  DIFFERENT instruments, and the gap is corpus-dependent (differential, not a
  constant offset). Re-drain the comparison pool on the same route before any
  cross-arm headline.
- Sync was also ~11x faster here (3.5 min vs 38.5 min for 9,000 calls) because
  it skips Batch queue latency — the batch-for-large-N guidance is a cost/TPM
  rule, not a latency one.

Related: [[feedback_batch_judge_aggregator_bare_int_parse]] (the sibling
bare-scalar trap one layer up, in the Betley aggregator),
[[feedback_rule24_surgical_rejudge_recipe]].

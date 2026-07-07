---
title: 'Realistic sparse-crossed prefix x query corpus: prefix-map vs context-map,
  averaging-rank collapse, and prefix/query variance decomposition of answer-state
  transport'
kind: experiment
tags: []
created_at: '2026-07-07T02:39:24Z'
has_clean_result: false
origin_prompt: 'how can we get more realistic and diverse contexts but also be able
  to compare the context vs query maps? potentially another issue is working on this
  [design discussion] -> Yes let''s run it in the background with happy coder. First
  ask clarifying questions [answers: all three co-primary; natural + 50-battery bridge;
  ~1k prefixes / ~13k rows; random + topic-matched control]'
workflow: v1
goal: 'Characterize how the base model''s answer-state transport decomposes between
  prefix and query on a realistic sparse-crossed corpus (~1k real WildChat/LMSYS prefixes
  + the #594 50-battery bridge, x ~500-query bank, natural + random + topic-matched
  pairings): co-primary reads = prefix-map vs context-map held-out R^2 + spectrum
  gap; #813''s averaged-vs-per-example rank collapse at un-capped design dimension;
  prefix/query/interaction variance shares of v(x).'
---
## Goal

Characterize how the base model's answer-state transport decomposes between prefix and query on a realistic sparse-crossed corpus (~1k real WildChat/LMSYS prefixes + the #594 50-battery bridge, x ~500-query bank, natural + random + topic-matched pairings): co-primary reads = prefix-map vs context-map held-out R^2 + spectrum gap; #813's averaged-vs-per-example rank collapse at un-capped design dimension; prefix/query/interaction variance shares of v(x).

1. **Prefix-map vs context-map.** Fit h_P: prefix-end activation -> v(x) and h_C: context-end (post-query) activation -> v(x) on the SAME rows; compare held-out R^2 (paired, group-level folds), full-output-space rank/energy spectra, and output-subspace overlap. Decision quantity: the paired held-out R^2 gap + spectrum gap between the two input representations — "how much transport does the query add". (The standing prefix+context both-arms rule is satisfied BY this read — the arm comparison is itself co-primary.)
2. **Query-averaged vs per-example grain at un-capped diversity.** Replicate the #813 rank comparison (averaged stable-rank 3.13 vs per-example 12.97 at L14; averaging ITSELF ~halves effective rank at matched n=50; median 2% output-energy overlap — eval_results/issue_813/per_example_vs_averaged/rank_spectrum_L14.json, script scripts/issue813_rank_spectrum.py) on a design whose dimension (~1.5k) no longer caps map rank. Does the averaging collapse persist, grow, or vanish when the design is realistic and wide?
3. **Variance decomposition of v(x)** into prefix main effects / query main effects / interaction, estimable because the random sparse crossing keeps the design graph connected. This is the fraction of answer-state transport that is prefix-driven — the quantity the condition-level map h(c_C) in the leakage-theory line (Overleaf paper, assumption A3 / a:context-vectorization) rides on.

**Corpus (clarified by user 2026-07-06):** ~1,000 real multi-turn prefixes (WildChat / LMSYS-1M conversations cut at the FINAL user turn — everything before the cut is a genuine prefix: real history, real task/persona setup; stratified by topic + length; toxicity-filtered) PLUS the #594 50-context battery verbatim as a bridge stratum (direct comparability with the #813/#594 line). Query bank: ~500 real final-turn user queries. Per prefix: (a) its own NATURAL query — the realism anchor, reproducing #779's regime; (b) ~12 RANDOMLY assigned shared queries from the bank — the primary identification arm (random incomplete crossing; prefix + query main effects identifiable); (c) a TOPIC-MATCHED crossed stratum (~+30% rows) as the pairing-realism control — does implausible prefix-query pairing distort the transport? (validates the crossing methodology, retroactively including #813's factorial). Total ~13k rows + control stratum.

**Capture:** Qwen/Qwen2.5-7B-Instruct (bf16), base model only, no adapters. Per row: one frozen greedy base response (vLLM batched); teacher-forced capture of prefix-end activation, context-end activation, and answer-span mean profile v(x), all 28 layers; reduced per-row summaries persisted (+ raw rollout text per upload policy — text uploads unconditionally). Est ~10 GPU-h (1x H100 class) — under the autonomous plan-approval cap.

**What would count as an answer:** (1) a quantified query-contribution to transport (R^2 gap + spectral gap, with CIs); (2) a verdict on whether question-averaging's rank collapse is a design artifact or a property of condition-level objects; (3) prefix/query/interaction variance shares with uncertainty; (4) a natural-vs-crossed distortion estimate that either licenses or caveats every crossed-design map result in the project.

**Competing hypotheses:** H-A: transport is mostly prefix-driven (high prefix share; small R^2 gap between h_P and h_C; averaging collapse persists) — supports condition-level h(c_C) as the right theory object. H-B: transport is mostly query-driven (large gap; per-example rank scales with query diversity; averaging collapse is information loss, not compression) — the condition-level object discards most of the map. #813's within-context R^2 0.71-0.87 (query-specific signal dominating held-out transfer) is prior evidence toward H-B at narrow diversity; #779's r_B-direction predictability (R^2 0.79-0.87) shows the trait read survives either way.

**Planner notes (grounding constraints, not prose):** group-level folds — hold out PREFIXES, never rows (ood-generalization-folds); lambda policy: shared-lambda primary across arms/grains (the #813 rank-spectrum precedent — kills differential shrinkage), per-grain GCV as sensitivity; reuse scripts/issue813_rank_spectrum.py factored-spectrum machinery + the #722-line Gram-dual ridge engine; fits are 0-GPU VM-CPU (vectorized, detached, checkpointed); frozen greedy responses teacher-forced once (no per-question judging needed for the three primary reads); WildChat/LMSYS prefix harvesting is tier-1/2 data realism; per-position/length decay reads are OWNED by #825 — out of scope here; new-direction literature review at plan time (crossed/factorial probing designs, relation-decoding linear maps, activation-transport literature) per the Critical Rules.

## Provenance

- Origin: user-chat design discussion 2026-07-06, immediately following the #813 inline free-analysis round `rank-spectrum-averaged-vs-per-example` (averaged vs per-example map rank, eval_results/issue_813/per_example_vs_averaged/) and the #779-vs-#813 corpus-diversity rank contrast (per-example map k90 >500 on 5000 free-form LMSYS prompts vs ~80-98 on the 50x<=64 factorial — map rank tracks design dimension).
- Clarifying answers (user, 2026-07-06): primary read = ALL THREE co-primary; corpus = natural + 50-battery bridge; scale = ~1k prefixes / ~13k rows first run; crossing = random primary + topic-matched control.

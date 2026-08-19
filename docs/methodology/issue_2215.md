# Methodology — issue 2215


**Design:** 39 pair-cells over 37 context-cells (21 base information types plus 6 context-load, 8 recency, and 4 instruction-vs-demonstration conflict variants), each with 12 carriers × 3 values → 1,404 contexts and 1,404 directed minimal pairs on Qwen-2.5-7B-Instruct. Per pair, exactly one information-type value changes; everything else is token-identical. Reads run at two slots (context-end = last context token; prefix-end = last token before the final user query) for three fitted ridge-map arms and two identity-plus-bias baselines, with no patching and no training. The four conflict pair-cells are forward/reverse twins re-pairing the same 72 contexts: under both-direction scoring each reverse statistic exactly duplicates its forward twin, so pooled counts weigh conflict pairs twice (37 effectively independent cells). Two cells are degenerate at prefix-end by design (the role-header and changed-question cells vary text after the prefix) and are excluded from all prefix-end reads — never drawn as zero bars. A zero-GPU follow-up round adds a completion-length covariate check on the answer-shift reads, using only the banked completion token counts (no new capture).

**Training:** N/A — no model training. Capture and analysis settings:

| Setting | Value | Source |
|---|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct, bf16 | run repro block, `dv3_map_discrimination.json` |
| Teacher-forced capture batch | 8 | parent capture default (`issue2162_run.py`), plan §4.2 |
| Capture rows | 14,040 = 1,404 contexts × 10 rollouts | Phase A coverage gate |
| Banked rollout recipe | 10 on-policy draws/context, temperature 1.0, seed 42 | parent anchors rows (verbatim fields) |
| Answer pooling | tail-inclusive mean (primary); span mean (secondary) | plan §4.3 |
| Layers | all 28 captured; map reads at 14/19/26; primary layer 19, frozen | plan §3 |
| Null / bootstrap draws | 10,000 each; seeds 2215 / 21620 | result JSON `meta` blocks |
| Map payloads | single-turn context-end ridge (LMSYS-fit, 963k rows); multi-turn prefix + context ridge (matched fit corpus and n) | plan §4.1, revision-pinned |
| Pooling-parity gate | cosine ≥ 0.995 on ≥ 99% of rows; realized min 0.999998, fraction 1.0 | pod parity manifest |
| Prefix-end token convention | index parity with the multi-turn map's convention confirmed on sampled contexts | pod A7 manifest, `all_match: true` |
| Length covariate (follow-up) | banked completion token counts, project tokenizer, retokenized; parity with the banked values asserted | `length_covariate.json` `length_metric` |
| LLM judge | none — geometry DVs only | plan §6 |

**Evaluation:** DV1 is the per-pair context-vector difference: magnitude as median norm against a carrier-shift yardstick (median distance between same-value contexts of different carriers — here, different final user queries), and direction consistency as the mean pairwise cosine of the difference across the 12 carriers, judged against a within-cell sign-flip permutation band (95th percentile, 10,000 draws). DV2 applies the same reads to the answer mean (teacher-forced states over the 10 banked rollouts, tail-inclusive pooling), with a split-half draw-noise yardstick. DV3 is paired two-alternative accuracy: per pair and direction, does the arm's prediction from the context vector sit closer in cosine to the correct answer mean than to the paired one — with carrier-clustered bootstrap CIs, a carrier-blocked shuffled-pair null, a leave-one-type-out identity-plus-bias baseline, nearest-neighbor retrieval over the 1,404-answer pool (chance 0.07% at k = 1), and pooled R²/cosine companions. The follow-up length-covariate check regresses per-pair answer-shift norm on the absolute difference in mean completion length (pooled ordinary least squares), re-reads the per-type medians and the coupling on the adjusted norms, and bootstraps raw and adjusted coupling on identical resample indices. Confirmation required, for the context-shift hypothesis, ratio > 1 and consistency above band for at least 18 of 21 base types; for the coupling hypothesis, Spearman ρ > 0.4 with an interval excluding 0; for the map-fidelity hypothesis, a pooled-accuracy interval above 0.5 and a fitted-minus-baseline interval above 0. Headline interval bounds (all carrier-clustered bootstrap, 10,000 draws):

| Quantity (layer 19, cosine, tail pooling) | Point | 95% interval | Verdict |
|---|---|---|---|
| Single-turn context-end map, pooled accuracy | 0.767 | 0.750 to 0.785 | discriminates |
| Multi-turn prefix-end map, pooled accuracy | 0.646 | 0.630 to 0.662 | discriminates |
| Single-turn context-end map − its baseline | +0.009 | −0.004 to +0.022 | inconclusive |
| Multi-turn context-end map − its baseline | +0.016 | +0.002 to +0.030 | beats baseline |
| Multi-turn prefix-end map − its baseline | +0.051 | +0.038 to +0.065 | beats baseline |
| Coupling with parent contrast (Spearman ρ, n = 38 cells) | +0.12 | −0.31 to +0.50 | inconclusive (spans 0 and 0.4) |
| Context-shift vs answer-shift coupling (ρ, n = 39 cells) | +0.64 | +0.45 to +0.75 | exploratory |
| Coupling, length-adjusted (ρ, n = 39 cells; paired bootstrap) | +0.56 | +0.34 to +0.69 | exploratory — survives adjustment |
| Answer shift vs completion-length difference (per-pair ρ, n = 1,404) | +0.35 | +0.30 to +0.40 | covariate present |

**Data extraction:** contexts, pairs, and both slot vectors come verbatim from the parent's frozen bank (1,404 contexts × 2 slots × 28 layers, fp32); the carriers are screened-benign WildChat-derived final user queries and the varied spans are hand-constructed minimal-pair values. Answer states are one unhooked teacher-forced forward per banked rollout (token-id concatenation, BPE-seam-safe), pooled over the answer span including the end-of-turn tail to match the maps' training-target convention; the span-mean twin matched the parent's banked store at ≥ 0.999998 cosine on all 14,040 rows. I acknowledge the fired conciseness WARNs — one Takeaways bullet exceeds the 30-word cap, several result blocks exceed the 120-word prose target, and the total prose exceeds the 800-word budget — nine result families (three dependent-variable families, their companion reads, the language-drift audit, and a length-covariate follow-up) are reported in one round; the cross-type heatmap is deliberately linked rather than embedded (one inline figure per result). <!-- concern-deferred: phase-d-no-entry-skip-sentinel --> One reviewer concern is deferred as repro hygiene: the run's finalize phase lacks an entry skip-if-done sentinel, so a re-entry would duplicate uploads and result markers; the phase ran once, and no reported measurement depends on it.

**Sample training/evaluation data + completions:** examples below are scored by the single-turn context-end map at layer 19; margins are cosine margins (positive = correct side). Full artifacts: [bank.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json) and the [banked rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/raw_completions/anchors) (16 jsonl shards).

Random draw (seed 42) from the 751 of 1,404 pairs correct in both directions — the prior-topic recency cell, 3 turns back, hiking vs birthday topic, carrier e10; all rows: [banked rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/raw_completions/anchors):

```
Varied span: conversation topic = "hiking" (v3) vs "birthday" (v1), 3 turns before the end
Final query (both sides): "Could you recap your main advice in one sentence?"
Rollout under hiking:   "For your three-day mountain hiking trip, plan your route and daily
                         distances carefully, secure accommodation in advance..."
Rollout under birthday: "Sure! For your daughter's 7th birthday party, consider a themed
                         treasure hunt, a craft corner, and fun games..."
Margins: +0.0562 / +0.0111 — both directions correct
```

Second random draw (seed 42) from the same correct-both-directions population — the knowledge-constraint cell, carrier e6; all rows: [banked rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/raw_completions/anchors):

```
System prompt: "You have no internet access: you cannot look anything up..." (v1)
           vs  "You have live web browsing and can look up current information..." (v2)
Final query (both sides): "How is the stock market doing right now?"
Rollout under v1: "I don't have real-time data access, so I can't provide the current
                   performance of the stock market. However..."
Rollout under v2: "I can help you check the current status of the stock market by looking
                   it up. To provide you with the most accurate..."
Margins: +0.0138 / +0.0079 — both directions correct
```

Cherry-picked, not a random sample — the single pair of 1,404 misclassified in both directions, the prior-topic recency cell at 5 turns back, birthday vs server-outage topic, carrier e9; all rows: [banked rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/raw_completions/anchors):

```
Varied span: conversation topic = "birthday" (v1) vs "outage" (v2), 5 turns before the end
Final query (both sides): "What's a common mistake people make with this?"
Rollout under v1: "It sounds like you're referring to designing a race car called the
                   Silver Centurion. However, if you're asking about..."
Rollout under v2: "When naming a race car like the Silver Centurion, a common mistake
                   people often make is focusing too much on the literal meaning..."
Margins: -0.0002 / -0.0008 — both directions wrong (near-ties)
```

By 5 turns back both answers orbit the most recent topic (a race-car naming exchange in the shared filler turns), so the paired answer means nearly coincide. Two further seed-42 draws from the 652 split pairs (one direction wrong): the numeric-list cell carrier e12 ("6 roses" vs "2 lanterns", margins −0.0011 / +0.0035) and the format-instruction cell carrier n8 (bullet points vs numbered steps, margins +0.0264 / −0.0115); their rollouts comply with the instructed format in both cases, and the missed directions again sit at near-zero margins.

---
title: Banked context-to-answer maps discriminate minimal-pair contexts, but at context-end
  an identity-plus-bias baseline captures most of the discrimination (MODERATE confidence)
kind: experiment
tags:
- followup-manual
created_at: '2026-08-10T05:01:23Z'
has_clean_result: true
parent_id: 2162
backend: runpod
workflow: v1
goal: 'On Qwen-2.5-7B-Instruct, using the frozen 21-type minimal-pair context bank
  and banked state bank from the parent patch sweep (#2162), measure with NO patching
  how changing exactly one information type in the context shifts (a) the context
  representation (context-end vector AND prefix-end state, both arms: per-type shift
  magnitude, within-type direction consistency, cross-type geometry) and (b) the realized
  on-policy answer representation v_A (teacher-forced answer activations over the
  banked floor/ceiling rollout text under each unpatched context), and (c) whether
  the banked context-to-answer ridge maps (#779 context-end, #1738 prefix-end; no
  new map training unless fitness fails) DISCRIMINATE the paired contexts — per pair,
  does f(v_C(A)) land closer to v_A(A) than to v_A(B) (paired two-alternative retrieval
  accuracy + cosine/euclidean similarity margin), reported per information type with
  a ranking of which types the map discriminates worst, alongside the identity+learned-bias
  baseline and pool-level kNN retrieval, against a shuffled-pair null.'
relates_to:
- spec-context-as-vector
---
# Banked context-to-answer maps discriminate minimal-pair contexts, but at context-end an identity-plus-bias baseline captures most of the discrimination (MODERATE confidence)

<!-- clean-result-v4 -->
**Methodology:** https://github.com/superkaiba/explore-persona-space/blob/057a76bd6aa265e212ed9982bba777ecd32c3241/docs/methodology/issue_2215.md · https://gist.github.com/superkaiba/a9aab98dd3ecfbf6f9e15e72f63ffee2 (post-fold re-export)


## Takeaways

- **0.77 pooled two-alternative accuracy** at context-end on both batteries: parent predictors pooled 0.59–0.77 (1,404 pairs); the new eight-type content battery reaches 0.77 against a 0.46–0.54 null (288 pairs), six of eight types individually.
- **+0.9 points** is all fitting adds over identity-plus-bias at context-end on the content battery (interval spanning zero; the baseline alone reaches 0.764) — the same 0.9-point gap the parent battery gave.
- **+9.3 points** is what fitting buys at prefix-end on content pairs, versus +0.5 at context-end; the slot contrast (+8.8 points) excludes zero. Parent round: +5.1 versus +0.9.
- **1.00 separability and 0.97 top-1 retrieval** put refusal-request first of the eight types on both axes, contradicting the hub-collapse prediction — a provisional falsification pending the refusal cell's 36-pair matching audit; weak types instead fail by value-generic predictions.
- **20 of 21 parent types** move the context-end vector consistently in direction but only 3 of 21 in magnitude; answer movement tracks context movement (ρ = 0.64), coupling with the parent's judge contrast undetected.
- Caveats: the ninth planned type, fact-truth, was dropped at the pair-validity gate (24 of 36 valid, floor 29); CJK drift (12.4% of parent, 5.8% of new rollouts) moves accuracies at most 2.8 points; the refusal cell's pair-matching human audit is pending (`xstest-human-audit-pending`); three re-entry upload-staleness concerns stay open (`dbe-hf-egress-content-staleness`, `dbe-percell-tensor-size-skip-staleness`, `dbe-d-upload-done-remote-staleness`) — every phase ran once, so no reported number consumed a re-entered upload.

## Goal

- **This experiment in context:** the parent [#2162](https://eps.superkaiba.com/tasks/2162) built the frozen 21-type minimal-pair context bank and measured which information types are causally usable under activation patching. This task asks, with no patching, how far changing one information type moves the context representation and the realized answer representation, and whether the banked context-to-answer ridge maps from [#779](https://eps.superkaiba.com/tasks/779) (single-turn, context-end) and [#1738](https://eps.superkaiba.com/tasks/1738) (multi-turn, prefix-end + matched context-end) preserve the pair distinction. A type the parent's probe decodes perfectly could still sit behind a map that cannot separate the paired answers — this is the map-fidelity read the parent could not give. A second follow-up round extends the same question to eight new content-level pair types (user identity, register, topic, language, sentiment, formatting, code-versus-prose, refusal-inducing requests), built non-overlapping with the parent's 21 types, and tests a dissociation prediction derived from [#2202](https://eps.superkaiba.com/tasks/2202)'s retrieval-failure hot-spots — measured there under a union-pool retrieval protocol, so the two retrieval reads are not directly comparable.
- **Broader narrative:** the context-to-answer map line asks whether pre-generation context geometry predicts realized answer representations. This experiment tests whether the fitted maps carry pair-level information beyond what raw context geometry plus a learned constant offset already provides — the floor any "the map understands the context" claim has to clear.

## Methodology

**Design:** 39 pair-cells over 37 context-cells (21 base information types plus 6 context-load, 8 recency, and 4 instruction-vs-demonstration conflict variants), each with 12 carriers × 3 values → 1,404 contexts and 1,404 directed minimal pairs on Qwen-2.5-7B-Instruct. Per pair, exactly one information-type value changes; everything else is token-identical. Reads run at two slots (context-end = last context token; prefix-end = last token before the final user query) for three fitted ridge-map arms and two identity-plus-bias baselines, with no patching and no training. The four conflict pair-cells are forward/reverse twins re-pairing the same 72 contexts: under both-direction scoring each reverse statistic exactly duplicates its forward twin, so pooled counts weigh conflict pairs twice (37 effectively independent cells). Two cells are degenerate at prefix-end by design (the role-header and changed-question cells vary text after the prefix) and are excluded from all prefix-end reads — never drawn as zero bars. A zero-GPU follow-up round adds a completion-length covariate check on the answer-shift reads, using only the banked completion token counts (no new capture). A second follow-up round (`discrimination-battery-expansion`) extends the battery to eight new content-level pair types on the same frozen model and maps: six constructed cells (12 carriers × 3 values → 36 contexts and 36 unordered pairs each — user role identity, style register, conversation topic, conversation language, user document format, code-versus-prose) and two benchmark-derived 36-item × 2-value cells (72 contexts, 36 pairs each) — refusal-request built from the XSTest safe/unsafe prompt bank (HF `Paul/XSTest`, `xstest_prompts.csv`, 450 rows — 250 safe, 200 unsafe; referenced here by filename and row counts only) and sentiment from IMDb contrast sets — for 360 contexts and 288 pairs, each scored in both directions. A ninth planned type, fact-truth, failed the pair-validity floor at data generation (24 of 36 judged-valid pairs, floor 29) and was dropped before any capture; every denominator below uses the realized 8 types. Fresh on-policy rollouts and teacher-forced captures ran on one H100; the maps are applied read-only, unchanged. The refusal-request and sentiment cells vary only the final user turn, so a mechanical prefix-token-identity check excludes them from every prefix-end read.

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
| New battery: bank | 8 types, 360 contexts, 288 pairs; per-cell complete grids asserted | `datagen_manifest_final.json` gate-1 block |
| New battery: datagen (drafting + pair-validity judge) | claude-sonnet-4-5-20250929 for both; drafting temperature 1.0, drafting `max_tokens` 8000, datagen/selection seed 2215; 324 sync pair-validity calls (one per grid pair across the nine planned types, reason-then-verdict, judge `max_tokens` 1024, malformed dropped never coerced) | `datagen_manifest_final.json` `reproducibility` + gate-1 block; plan v6 §4.2/§11 |
| New battery: rollouts | 10 draws/context, temperature 1.0, seed 42; 3,600 rows; `max_new_tokens` 2048; generation batch 16, capture batch 8 | run sentinel + plan v6 §4.3 |
| New battery: cap-hit fraction | 5 of 3,600 rows (0.14%), basis = retokenized completion length at the cap; 2% per-cell re-generation trigger — no cells fired | upload-verification PASS note |
| New battery: capture-parity gate | 6 independent re-capture checks, 0 failures; span keys cosine ≥ 0.9999; single-position context keys ≥ 0.999 (early layers) / ≥ 0.995 (flattened), bf16 two-bar calibration | gate-4 manifest; calibration commit `62c686d845` |
| New battery: nulls / bootstrap | within-cell derangement null + cluster bootstrap, B = 10,000 each, seeds 2215 / 21620; pooled reads use equal type weights | `dv3_dbe_map_discrimination.json` `meta` |
| New battery: judge (refusal manipulation check) | claude-sonnet-4-5-20250929, sync, `max_tokens` 1024, drop-never-coerce; 720 judged, 0 content drops, 0 transport losses | `refusal_manipulation_check.json` |
| LLM judge (geometry DVs) | none — geometry DVs only | plan §6 |

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

The `discrimination-battery-expansion` round applies the same discrimination machinery to the new bank at the same frozen layer 19 (cosine, tail-inclusive targets; layers 14/26, euclidean, and span-mean reported in full as sensitivity reads): per-type paired two-alternative accuracy with within-cell derangement nulls and carrier- or item-clustered bootstrap intervals (B = 10,000 each), pooled as the unweighted mean over types (each bootstrap draw resamples clusters within type before averaging types, so 36-item benchmark cells do not outweigh 12-carrier constructed cells); per-(type × arm) transfer R², mean cosine, and retrieval acc@{1, 5, 10} over the 360-context pool (chance 0.28% at k = 1); and a slot contrast — the prefix-end fitted-minus-baseline gain minus the context-end gain on the matched-fit map pair, computed over the six prefix-varying types inside every bootstrap draw. Decision criteria over the realized type count: broad discrimination supported at 7 or more individually-discriminating types of 8 and falsified at 3 or fewer (realized 6 — inconclusive); context-end near-equivalence supported when the pooled fitted-minus-baseline upper bound is at or under +0.03 (realized upper bound +0.028 — supported); the slot contrast supported when its interval excludes zero from below (realized — supported); the refusal dissociation prediction falsified if refusal retrieval sits above the type median (realized — falsified, provisional pending the refusal cell's pair-matching audit). New-battery interval bounds (layer 19, cosine, tail pooling, equal type weights):

| Quantity (new battery) | Point | 95% interval | Verdict |
|---|---|---|---|
| Single-turn context-end map, pooled accuracy (8 types) | 0.773 | 0.760 to 0.786 | discriminates |
| Multi-turn context-end map, pooled accuracy (8 types) | 0.773 | 0.759 to 0.786 | discriminates |
| Multi-turn prefix-end map, pooled accuracy (6 prefix-varying types) | 0.664 | 0.641 to 0.688 | discriminates |
| Identity-plus-bias baseline, context-end (8 types) | 0.764 | 0.748 to 0.781 | discriminates |
| Identity-plus-bias baseline, prefix-end (6 prefix-varying types) | 0.572 | 0.551 to 0.595 | discriminates |
| Single-turn context-end map − its baseline (8 types) | +0.009 | −0.010 to +0.028 | near-equivalent (upper bound under +0.03) |
| Multi-turn prefix-end map − its baseline (6 prefix-varying types) | +0.093 | +0.067 to +0.118 | beats baseline |
| Slot contrast: prefix-end gain − context-end gain (matched fit) | +0.088 | +0.058 to +0.120 | prefix-end gains more from fitting |
| Refusal-request: two-alternative accuracy / retrieval acc@1 | 1.00 / 0.97 | rank 1 of 8 types on both axes | hub-collapse prediction falsified (provisional pending pair audit) |

**Data extraction:** contexts, pairs, and both slot vectors come verbatim from the parent's frozen bank (1,404 contexts × 2 slots × 28 layers, fp32); the carriers are screened-benign WildChat-derived final user queries and the varied spans are hand-constructed minimal-pair values. Answer states are one unhooked teacher-forced forward per banked rollout (token-id concatenation, BPE-seam-safe), pooled over the answer span including the end-of-turn tail to match the maps' training-target convention; the span-mean twin matched the parent's banked store at ≥ 0.999998 cosine on all 14,040 rows.

New-battery contexts are freshly constructed: the six constructed cells are diverse synthetic (tier 3) drafted by claude-sonnet-4-5-20250929, seeded on WildChat query styles and real multi-turn conversations. The same model judged every candidate pair for validity — 324 sync calls, one per grid pair across the nine planned types, reason-then-verdict, malformed returns dropped never coerced — and the bank was frozen before any pod work. New-battery answer states persist both poolings per row (tail-inclusive primary, span-mean secondary).

The two benchmark cells come from published banks. Sentiment uses IMDb contrast sets — the 36 shortest reviews with human minimal edits flipping polarity. Refusal-request uses the XSTest prompt bank — 36 unsafe prompts greedily matched to their most lexically-similar safe siblings; the planned human audit of the 36 matches is pending, a carried scope caveat.

I acknowledge the fired conciseness WARNs — some Takeaways bullets exceed the 30-word cap, several result blocks exceed the 120-word prose target, and the total prose exceeds the budget — sixteen result families across the original run and two follow-up rounds are reported in one body. The cross-type heatmap is deliberately linked rather than embedded (one inline figure per result).

Figure disposition for the new round: `dbe_percell_2afc.png` and `dbe_h3_dissociation.png` are superseded by the embedded `dbe_hero_pertype_2afc.png` and `dbe_2afc_vs_retrieval.png`, and `dbe_joint_taxonomy.png` by `dbe_joint_taxonomy_48.png` (the filename says 48; the realized joint taxonomy has 47 cells). Committed but not embedded, each with its load-bearing read quoted in the results: `dbe_carrier_transfer.png` and `dbe_parent_fit_offset.png` (own-versus-cross accuracies and the parent-fit offset check — deliberately linked from their results rather than embedded), `dbe_length_covariate.png` (type-level rank correlation between context-length difference and accuracy is −0.01), `dbe_pooling_twins.png` (the largest per-type tail-versus-span accuracy delta is 4.2 points — a pooling-sensitivity read), `dbe_dv12_geometry.png` (exploratory per-cell shift and consistency reads; values in the dv1/dv2 JSONs), and `dbe_union_pool_retrieval.png` (protocol cross-reference to the earlier union-pool read — joint 1,764-context pool; numbers in `exploratory_dbe.json`). The per-pair margin panel `dbe_perpair_margins.png` was re-rendered with plain-English labels in a later revision round: its per-pair rows recompute from the persisted per-arm prediction matrices and targets (`issue2215_dbe/analysis_tensors/predictions/`) via `scripts/issue2215_dbe_perpair_recompute.py`, and every recomputed point matches the pod-side render's embedded data within 0.00012 (the numeric round-trip of the half-precision stored tensors).

An earlier deferred reviewer concern — the parent run's finalize phase lacked an entry skip-if-done sentinel — was addressed during this round (the finalize phase now carries a done-sentinel entry guard). Three reviewer concerns from the new round are deferred as repro hygiene: <!-- concern-deferred: dbe-gate4-metadata-independence --> the capture-parity gate's exact tail-metadata comparison shares wrapper bookkeeping with the generation loop rather than an independent capture-emitted record (its cosine checks are independent); <!-- concern-deferred: dbe-fabricated-force-resume-coverage --> forced failure-and-re-entry tests cover the analysis phase end-to-end but not the generation and finalize phases; <!-- concern-deferred: dbe-d-stale-sentinel-completion --> the finalize phase writes its done-sentinel before the last manifests upload, so a re-entered finalize could compose a stale sentinel. Each phase of the new round executed once, end-to-end, and no reported measurement depends on re-entry behavior.

**Sample training/evaluation data + completions:** examples below are scored by the single-turn context-end map at layer 19; margins are cosine margins (positive = correct side). Full artifacts: [bank.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json) and the [banked rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo/raw_completions/anchors) (16 jsonl shards); new-battery rollouts under [issue2215_dbe/raw_completions/anchors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/60d7014b93d4889f0358680ded3126474cd312f4/issue2215_dbe/raw_completions/anchors) (8 per-type jsonl files).

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

From the new battery — random draw (seed 42) from the 34 of 36 conversation-topic pairs correct in both directions, carrier c03, retirement-account troubleshooting vs houseplant care realizing the same conversation skeleton; all examples: [qualitative_examples.json](https://github.com/superkaiba/explore-persona-space/blob/488b3e3cee4629427d484c35d8674f63a3bdf41a/eval_results/issue_2215/discrimination-battery-expansion/qualitative_examples.json):

```
Varied span: the whole conversation's domain — 401(k) growth troubleshooting (v2)
             vs fiddle-leaf-fig leaf drop (v3); same skeleton, topic-generic final query
Final query (both sides): "Could you recap your main advice in one sentence?"
Rollout under v2: "Log into your 401(k) account portal to review your investment
                   allocations and employer match, and consider shifting a majority..."
Rollout under v3: "Move your fiddle leaf fig to a bright, indirect light spot near a
                   window and water it when the soil is dry to a depth of about two inches."
Margins: +0.0712 / +0.0469 — both directions correct
```

Second new-battery random draw (seed 42) from the 5 of 36 user-role-identity pairs correct in both directions — the same used-car question opened by a different self-introduction, carrier c04; margins two orders of magnitude smaller than the topic cell's; all examples: [qualitative_examples.json](https://github.com/superkaiba/explore-persona-space/blob/488b3e3cee4629427d484c35d8674f63a3bdf41a/eval_results/issue_2215/discrimination-battery-expansion/qualitative_examples.json):

```
Varied span: first-turn self-introduction — "Hi, I'm an emergency-room nurse." (v1)
             vs "Hi, I'm a criminal-defense lawyer." (v2); everything else identical
Final query (both sides): "If you had to pick just one thing that matters most,
                           what would it be?"
Rollout under v1: "If you had to pick just one thing, a thorough pre-purchase inspection
                   by an independent mechanic is critical..."
Rollout under v2: "If you had to pick just one thing, a comprehensive vehicle history
                   report is crucial. This report, often obtained through services..."
Margins: +0.00007 / +0.00068 — both directions correct, near-zero margins
```

Random draw (seed 42) from the 36 of 36 conversation-language pairs correct in both directions — the same Tomcat servlet-filter debugging exchange in English vs French, item c03; all examples: [qualitative_examples.json](https://github.com/superkaiba/explore-persona-space/blob/488b3e3cee4629427d484c35d8674f63a3bdf41a/eval_results/issue_2215/discrimination-battery-expansion/qualitative_examples.json), full per-type rollouts: [anchors tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/60d7014b93d4889f0358680ded3126474cd312f4/issue2215_dbe/raw_completions/anchors):

```
Varied span: the conversation's language — English (v1) vs French (v3);
             same 5-message debugging exchange
First user turn (v1): "why if i put link of an image to .html file it cant be read
                       in tomcat if i have filter"
First user turn (v3): "pourquoi si je mets un lien d'une image dans un fichier .html
                       il ne peut pas être lu dans tomcat si j'ai un filtre"
Rollout under v1: "Yes, you can configure a `filter-mapping` in `web.xml` to exclude
                   specific URLs or directories from being processed by your filter..."
Rollout under v3: "En utilisant `web.xml`, vous pouvez exclure des chemins spécifiques
                   en configurant des `filter-mapping` séparés..."
Margins: +0.0120 / +0.0259 — both directions correct
```

Random draw (seed 42) from the 35 of 36 review-sentiment pairs correct in both directions — a single-turn draft-review request whose IMDb review carries three minimal human edits flipping polarity, item i08; all examples: [qualitative_examples.json](https://github.com/superkaiba/explore-persona-space/blob/488b3e3cee4629427d484c35d8674f63a3bdf41a/eval_results/issue_2215/discrimination-battery-expansion/qualitative_examples.json):

```
Varied span: the review's polarity — "...abstract images, are enough to drive anyone
             crazy" (v1) vs "...are enough to drive anyone jumping on their seats" (v2),
             plus two sibling clause edits ("just not"→"definitely", "too long"→"of joy")
Query (both sides): the same "Here's my draft review: ..." single-turn request
Rollout under v1: "Based on your review, it seems that you didn't particularly enjoy
                   the film, finding it too long and not to your taste..."
Rollout under v2: "Based on your enthusiastic review, it sounds like you thoroughly
                   enjoyed the experience..."
Margins: +0.0068 / +0.0165 — both directions correct
```

Random draw (seed 42) from the 8 of 36 user-document-format pairs correct in both directions — a library book inventory pasted as a markdown table vs the same records as JSON, item c02; all examples: [qualitative_examples.json](https://github.com/superkaiba/explore-persona-space/blob/488b3e3cee4629427d484c35d8674f63a3bdf41a/eval_results/issue_2215/discrimination-battery-expansion/qualitative_examples.json):

```
Varied span: the pasted document's format — markdown table (v1) vs JSON records (v2);
             identical content and query
Final query (both sides): "How many total copies does the library have of books
                           written by Marina Wells?"
Rollout under v1: "To determine the total number of copies of books written by Marina
                   Wells in the library, let's identify the relevant entries..."
Rollout under v2: "To find out how many total copies the library has of books written
                   by Marina Wells, we need to sum up the copies for all books..."
Margins: +0.00005 / +0.0026 — both directions correct, the a-direction barely positive
```

Random draw (seed 42) from the 3 of 36 code-versus-prose pairs correct in both directions — the same retry-with-backoff logic as pseudocode vs a prose description, item c09; all examples: [qualitative_examples.json](https://github.com/superkaiba/explore-persona-space/blob/488b3e3cee4629427d484c35d8674f63a3bdf41a/eval_results/issue_2215/discrimination-battery-expansion/qualitative_examples.json):

```
Varied span: presentation of the same algorithm — "FUNCTION retry_with_backoff(operation,
             max_attempts, base_delay): FOR attempt FROM 0..." (v2, pseudocode)
             vs "The retry mechanism attempts to execute an operation up to a maximum
             number of times, wi..." (v3, prose)
Final query (both sides): "What's one improvement you'd suggest?"
Rollout under v2: "One improvement to the `retry_with_backoff` function could be to
                   introduce randomness into the delay between retries..."
Rollout under v3: "One improvement I would suggest is to include a maximum delay
                   constraint to prevent the retry mechanism from waiting indefinitely..."
Margins: +0.0036 / +0.0059 — both directions correct
```

Cherry-picked, not a random sample — style-register has 0 of 36 pairs correct in both directions, so the drawn example is its worst miss, pair c07 (a formal vs casual request for the same 60 coloring-book prompts); all examples: [qualitative_examples.json](https://github.com/superkaiba/explore-persona-space/blob/488b3e3cee4629427d484c35d8674f63a3bdf41a/eval_results/issue_2215/discrimination-battery-expansion/qualitative_examples.json):

```
Varied span: register of the same request — "List of 60 prompts to create Space
             exploration coloring book images for toddlers" (v2, formal)
             vs "hey can you give me like 60 prompts for space exploration coloring
             book pics for toddlers" (v3, casual)
Rollout under v2: "Yes, that's a great idea. Each prompt should clearly specify that
                   the images should be in black and white line art with no shading..."
Rollout under v3: "Got it! Here's a structured list of 60 simple space exploration
                   prompts for a coloring book..."
Margins: -0.0179 / +0.0174 — one direction wrong, near-mirror margins
```

Refusal-request example (structural description only — bank rows are referenced by filename and index, never quoted, for content hygiene): the seed-42 draw from the 36 of 36 pairs correct in both directions is pair i08, a single-turn cell whose only varied slot is the sole user message (an unsafe XSTest prompt, 35 characters, vs its lexically-matched safe sibling, 34 characters). Its margins are +0.3281 / +0.3259 — an order of magnitude larger than any constructed cell's — and the judged refusal rates behind the cell are 84.4% on the unsafe side vs 13.6% on the safe side. Rows: [anchors_dbe_w0_refusal_request.jsonl](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/60d7014b93d4889f0358680ded3126474cd312f4/issue2215_dbe/raw_completions/anchors/anchors_dbe_w0_refusal_request.jsonl), indexed by `pair_id`.

## Results

### Maps discriminate paired answers well above chance, but a constant offset does most of it at context-end

Per information type, paired two-alternative accuracy at layer 19 (cosine; tail-inclusive targets from teacher-forced re-forwards of the banked on-policy rollouts): four headline predictors plus per-type shuffled-pair null bands (gray); 36 pairs per cell, worst types at top.

![Per-type paired two-alternative accuracy at layer 19 for the single-turn context-end map, multi-turn prefix-end map, multi-turn context-end map, and identity-plus-bias baseline, with shuffled-pair null bands](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/hero1_per_type_2afc.png)

> **Figure.** *Discrimination is broad but heterogeneous.* Per-type paired two-alternative accuracy, four predictors plus per-type shuffled-pair null bands (gray), 36 pairs per cell, worst types at top. Prefix-end bars are omitted for the two cells degenerate at that slot.

Pooled: single-turn context-end 0.767, multi-turn context-end 0.774, prefix-end 0.646; identity-plus-bias 0.758 (context-end), 0.595 (prefix-end). All clustered intervals clear the 0.48–0.52 pooled null; each fitted context-end map discriminates in 34 of 39 pair-cells (interval above chance; baseline 31). Fitted-minus-baseline: +0.9 points single-turn context-end (interval includes zero), +1.6 multi-turn context-end, +5.1 prefix-end (intervals exclude zero; Methodology table); the slot claim rests on the matched multi-turn pair only.

Worst types: the user-name and prior-topic recency/load cells and demo-format (0.49–0.58); on some instruction-heavy cells the baseline beats the fitted single-turn map (knowledge-constraint 0.903 vs 0.708). Single-model, correlational, no-patching reads; answer-pooling twins move the fitted maps at layer 19 by ≤ 0.6 points (baselines up to 1.1); accuracy peaks at layer 19.

### The low-level view: misses are near-ties, not confident inversions

Per-pair cosine margins per type for the three fitted arms (each dot one scored direction of one pair; dashed line = zero margin; same worst-to-best order as above).

![Per-pair cosine margins per information type for the three fitted map arms, with a dashed zero line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/expl_margin_scatter_per_type.png)

> **Figure.** *Per-pair margins behind the accuracy bars.* Each dot is one direction of one minimal pair (2,808 per context-end panel; 2,664 in the prefix-end panel, whose two degenerate cells are excluded); margins right of the dashed zero line are correct calls. Weak types cluster tightly around zero rather than inverting.

For the single-turn context-end map, 751 of 1,404 pairs are correct in both directions, 652 in exactly one, and 1 in neither; missed directions sit at margins within about 0.01 of zero. Weak types fail by indistinguishability of the paired answer means, not by systematic inversion.

### Retrieval and fit companions dissociate: the prefix-end map discriminates pairs it cannot retrieve

Nearest-neighbor retrieval of each context's true answer mean among all 1,404, per arm: fraction of contexts whose true answer mean is within the k nearest neighbors of the prediction (k = 1, 5, 10; chance 0.07%, 0.36%, 0.71%), cosine and euclidean panels.

![Nearest-neighbor retrieval accuracy at k equals 1, 5, and 10 for all five predictors, cosine and euclidean, with chance levels marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/expl_knn_retrieval.png)

> **Figure.** *Pool-level retrieval separates the arms far more than pair discrimination does.* Context-end arms retrieve the true answer mean in the top 10 of 1,404 for 69–82% of contexts; prefix-end arms sit near 1–9%.

Context-end maps reach 34–35% top-1 (median rank 2–3) with pooled R² 0.57–0.61; the prefix-end map, still discriminating pairs at 0.646, retrieves 1.1% top-1 (median rank 287) with pooled R² −0.13; the identity-plus-bias baselines hold negative R² everywhere while discriminating at up to 0.758 — fit and discrimination dissociate in both directions. Two structural notes bind the absolute numbers: prefix-end predictions take at most 3 distinct values per cell (shared prefixes), capping pool-level retrieval; and absolute R²/cosine carry the capture-convention seam between the maps' training rig and this bank, so the paired within-bank statistics are the seam-protected reads.

### Context-end discrimination is partly pair-specific; prefix-end discrimination is entirely value-generic

Per type and arm, own-pair accuracy against accuracy when the same predictions are scored against another carrier's same-value answer means (cross-carrier targets); the diagonal marks no own-pair advantage.

![Own-pair accuracy versus cross-carrier accuracy per type for all five predictors, with a diagonal reference line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/expl_carrier_transfer.png)

> **Figure.** *How much of the discrimination survives swapping the target carrier.* Context-end arms sit below the diagonal (own-pair advantage); both prefix-end arms sit exactly on it, as forced by their carrier-identical predictions.

Pooled own-vs-cross: single-turn context-end 0.767 vs 0.677, its baseline 0.758 vs 0.710, prefix-end arms exactly equal by construction. Most context-end discrimination transfers across carriers (value-generic), but the fitted map's own-pair advantage (9.0 points) exceeds the baseline's (4.8), so the fitted map carries some genuinely pair-specific structure beyond a generic re-alignment. The changed-question control is the extreme case: the fitted map's own-pair 0.972 collapses to 0.484 — chance — on cross-carrier targets (0.958 to 0.496 for the baseline), so its discrimination is entirely pair-specific, exactly as its query-specific shift direction predicts.

### One information-type change moves the context-end vector by less than swapping the final user query

Per type, median pair-shift magnitude divided by its yardstick, log scale: context vectors at both slots against the cross-carrier same-value distance, answer vectors against the split-half draw-noise floor; dashed line at ratio 1. Prefix-end points appear only for the 5 cells with a nonzero yardstick.

![Per-type shift magnitude relative to its yardstick on a log scale, for context vectors at both slots and answer vectors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/hero2_shift_ratio_per_type.png)

> **Figure.** *Magnitude ratios per type.* Context-end ratios (orange) mostly sit below 1; answer-vector ratios (black) mostly above. The near-absent prefix-end series reflects a yardstick that is zero by construction for 32 of 37 cells.

At context-end the yardstick is a swap of the final user query — the span nearest the read slot — and one prefix-side type change moves the vector 0.05–1.6× that: 3 of 21 base types exceed 1 (in-context-learning mapping 1.58, format instruction 1.38, prompted persona 1.38). The context-shift hypothesis' magnitude criterion (18 of 21) fails decisively; granting the two base types within 0.03 of threshold still gives 5 of 21. The prior-topic ratio falls from 0.55 (base) to 0.39 (3 turns back) to 0.16 (5 turns back), so context-end magnitude rankings partly encode span position, not only information content.

### Direction consistency is high at context-end; the prefix-end consistency read is vacuous by construction

Per type, direction consistency (mean pairwise cosine of the pair-shift across the 12 carriers, clustered 95% CIs) against a within-cell sign-flip permutation band (95th percentile, gray ticks), for context vectors at both slots and answer vectors.

![Within-type direction consistency with permutation-null bands, three panels: context-end context vectors, prefix-end context vectors, and answer vectors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/expl_consistency_vs_band.png)

> **Figure.** *Directions are reliable at context-end.* Context-end consistency (orange) clears the null band for 38 of 39 pair-cells; the prefix-end panel reads exactly 1.0 everywhere because all carriers of a cell share one prefix.

Context-end consistency spans 0.15–0.83, clearing the band for 20 of 21 base types; the exception, the changed-question control (−0.01 against a 0.05 band), is query-specific by design. Shift directions are type-specific, not a generic "something changed" direction: mean absolute cosine is 0.482 within cells vs 0.101 between cells ([heatmap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/expl_cross_type_cosine_heatmaps.png)). This split and the changed-question control bound a lexical-propagation alternative — consistent directions carried by the literal inserted value tokens propagating to the read slot rather than by encoded content — without eliminating it (carriers of a cell share that text).

The prefix-end panel carries no information: with token-identical prefixes, every carrier's shift is the same vector, so consistency is identically 1 against the 0.18 sign-flip band; per-unit tail p-values below 1/4096 are unavailable (12-carrier sign-flip atoms).

### Answer-vector movement tracks context-vector movement; coupling with the parent's judge contrast is undetected

Left: per-cell noise-normalized answer-vector shift against the parent's per-cell behavioral separation (judge-scored ceiling-minus-floor contrast), points labeled. Right: per-type mean discrimination margin against the same x-axis for the two goal-named maps.

![Answer-vector shift and discrimination margin plotted against the parent behavioral separation per cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4dcde9ce0eac58ce16ae10b0827a98bc46401a71/figures/issue_2215/expl_h2_shift_vs_separation.png)

> **Figure.** *The coupling read.* Per-cell noise-normalized answer-vector shift vs the parent's judge contrast (n = 38 cells; the filler-swap cell has no parent rows; interval bounds in the Methodology table). The in-context-learning cell is the extreme outlier at 49.7.

The coupling hypothesis fails its confirmation criterion: ρ = 0.12 (n = 38) with an interval spanning zero and the ρ = 0.4 confirmation threshold — no detectable rank coupling, and the data cannot establish decoupling either; the conflict forward/reverse twins double four points. Raw answer-vector movement, in contrast, rank-tracks raw context-end movement across cells (ρ = 0.64, interval excluding zero, n = 39) and within cells (median per-cell ρ = 0.64). 30 of 39 pair-cells shift above the split-half noise floor; the 9 below it (demo format, demo persona, filler swap, user emotion, role header, user-name recency at 3 and 5 turns back, user-name load at 5 items, prior-topic at 5 turns back) are below-noise-floor reads — the floor is conservative by roughly √2, so "no shift" is not established for them.

### Completion-length differences covary with answer-vector shift but do not carry the per-type ranking or the coupling

Per pair, the answer-shift norm against the absolute difference in mean completion length (banked token counts; n = 1,404), plus per-type medians before and after a pooled linear length adjustment and the coupling recomputed on adjusted medians.

![Per-pair answer shift versus completion-length difference; per-type medians and coupling, raw versus length-adjusted](https://raw.githubusercontent.com/superkaiba/explore-persona-space/52c7ed13492bf8a12c4f6843a03f3a70c525c6c0/figures/issue_2215/fu_length_covariate.png)

> **Figure.** *Length-covariate check on the answer-shift reads.* Top left: per-pair scatter with the pooled fit line. Top right: per-type medians before vs after length adjustment; points near the dashed diagonal keep their rank. Bottom: the coupling on raw (left) and length-adjusted (right) medians; interval bounds in the Methodology table.

Pairs whose completions differ more in length shift more: pooled per-pair ρ = 0.35 (n = 1,404), yet the per-cell median is 0.14 (27 of 39 cells positive) — most of the pooled association is between-type. Removing the pooled length trend leaves the per-type ranking essentially unchanged (rank stability 0.94): the answer-shift ranking is not carried by length. The coupling with context-end movement drops from 0.64 to 0.56 under a paired bootstrap, its interval still excluding zero (Methodology table).

Eight of the nine below-noise-floor cells from the previous result (all but the filler swap) move just above the floor once shifts are put on an equal-length footing (adjusted ratios 1.01–1.37). The yardstick itself is not length-adjusted, so these are marginal equal-footing reads, not new detections.

### Language-drift audit: 12.4% of banked rollouts contain CJK text; headline reads are unchanged without them

Pooled paired two-alternative accuracy per arm (layer 19, cosine), computed from all 10 draws per context (filled, with carrier-clustered CIs) and recomputed with every CJK-bearing rollout excluded from the answer means (open); gray band is the shuffled-pair null.

![Pooled discrimination accuracy per arm with all draws versus with CJK-intruded draws excluded](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215/audit_cjk_recount.png)

> **Figure.** *Intrusion-sensitivity recount.* Filled dots: committed pooled accuracy (all 10 draws, clustered 95% CIs). Open dots: recount excluding the 1,736 CJK-bearing rollouts. The two coincide to within 0.4 points for every arm.

1,736 of 14,040 banked rollouts (12.4%) contain CJK characters (median intruded row: 1.7% of characters), concentrated in the implied-language (43.9% of the cell's rollouts) and language-instruction (28.1%) cells — whose values instruct Spanish, English, or French, so CJK is never on-task — though the tail is broad (filler swap 25.3%; several recency and demonstration cells near 20%). The all-draws recount reproduces the committed accuracies exactly (delta 0); excluding intruded draws changes pooled accuracy by at most +0.4 points and moves no per-cell accuracy across its null band (a point-versus-band check; per-cell recounts cover the single-turn context-end and prefix-end maps), and the largest per-type change is implied-language at prefix-end improving from 0.736 to 0.861 — drift adds target noise rather than manufacturing discrimination. No context loses all 10 draws.

### The eight-type content battery discriminates broadly but heterogeneously: six of eight types clear chance, two sit at it

Per new information type, paired two-alternative accuracy at layer 19 (cosine, tail-inclusive targets) for the three fitted maps and both identity-plus-bias baselines, with per-type shuffled-pair null bands; 36 pairs per type, worst types left.

![Paired two-alternative accuracy per new information type for five predictors with shuffled-pair null bands](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d7d3d23fbefdf4f72adbb8b7822968518f17a22/figures/issue_2215/dbe_hero_pertype_2afc.png)

> **Figure.** *Six of eight new content types are discriminated above chance.* Paired two-alternative accuracy per type, five predictors, shuffled-pair null bands (gray); daggers mark the two benchmark-derived cells; prefix-end bars are absent for the two cells whose pair sides share the prefix.

Pooled over the eight types the fitted context-end maps reach 0.773 and the prefix-end map 0.664, against a 0.46–0.54 pooled null; style-register (0.50) and code-versus-prose (0.54) are inconclusive, so the broad-discrimination criterion (support at 7 or more of 8) lands inconclusive at 6. The two benchmark cells — refusal-request and sentiment — sit at or near 1.0, so the LLM-constructed cells are not systematically easier than benchmark-derived ones.

The ninth planned type, fact-truth, was dropped at the pair-validity gate (24 of 36 valid, floor 29) before capture, so it appears in no figure. Per-type context-length differences do not track discrimination (type-level rank correlation −0.01).

### Weak new types fail by value-generic predictions: 0 to 8 of 36 pairs correct both ways

Per new type, each pair's two directed cosine margins for the single-turn context-end map — one point per pair, the two axes the two scoring directions, positive = correct call.

![Per-pair cosine margins for the two scoring directions of each new-battery pair, one panel per type](https://raw.githubusercontent.com/superkaiba/explore-persona-space/40cc2bb752c70d6e919c9ce5f1c1aa11f9e8b531/figures/issue_2215/dbe_perpair_margins.png)

> **Figure.** *Two failure regimes.* Weak types line up on the anti-diagonal — one direction's gain is the other's loss — while strong types fill the both-correct quadrant; daggers mark benchmark cells.

In the four weak types the two directions' margins are near mirror images: the predictions barely move between the pair sides, so whichever target sits nearer wins one direction and loses the other (0, 3, 5, and 8 of 36 pairs correct both ways for style-register, code-versus-prose, user-role-identity, and user-doc-format; strong types 34–36 of 36). A carrier-transfer read confirms the value-generic failure: own-pair and cross-carrier accuracy coincide for the weak types (style-register 0.500 versus 0.506, code-versus-prose 0.542 versus 0.545) while refusal-request keeps a 26-point own-pair advantage (1.000 versus 0.740) — [per-type transfer view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d7d3d23fbefdf4f72adbb8b7822968518f17a22/figures/issue_2215/dbe_carrier_transfer.png).

### The constant-offset result replicates on content pairs: +0.9 points at context-end, +9.3 at prefix-end

Fitted-minus-baseline accuracy gain at each read slot for the matched-fit multi-turn map pair, and their difference (the slot contrast), over the six prefix-varying types with clustered 95% intervals.

![Fitted minus baseline accuracy gain at prefix-end and context-end for the matched-fit maps and their difference](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d7d3d23fbefdf4f72adbb8b7822968518f17a22/figures/issue_2215/dbe_did_slot_gains.png)

> **Figure.** *Fitting matters at prefix-end, not at context-end.* Prefix-end gain +9.3 points, context-end gain +0.5, slot contrast +8.8 — the two gain legs share one fit corpus and n and differ only in read slot.

![Per-type companion of the slot-gain contrast, with pooled bars and the single-turn leg](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d7d3d23fbefdf4f72adbb8b7822968518f17a22/figures/issue_2215/dbe_slot_gains_did.png)

> **Figure.** *The per-unit companion behind the aggregate contrast.* Left: per-type fitted-minus-baseline gains at each slot over the six prefix-varying types. Right: pooled gains, the slot contrast, and a descriptive single-turn leg (+9.0 points) matching the matched-fit contrast.

At context-end the fitted maps again add almost nothing over identity-plus-bias: +0.9 points pooled over all eight types with an interval spanning zero, meeting the near-equivalence criterion (upper bound +2.8 under the +3-point bar; the baseline alone reaches 0.764). At prefix-end the same-fit map gains +9.3 points, and the slot contrast (+8.8, interval excluding zero; Methodology table) confirms that fitting buys discrimination specifically where the read slot sits far from the varied text — the parent round's +5.1 versus +0.9 pattern, widened. Re-fitting the baseline's offset on the parent battery's 1,404 pairs moves no new type by more than 1.4 points ([battery-internal versus parent-fit offset](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d7d3d23fbefdf4f72adbb8b7822968518f17a22/figures/issue_2215/dbe_parent_fit_offset.png)).

### Refusal pairs are separable (1.00) and best-retrieved (0.97 top-1): the hub-collapse prediction is provisionally falsified

Per new type, paired two-alternative accuracy against nearest-neighbor retrieval of the true answer mean at k = 1 (cosine, pool = all 360 new-battery contexts, chance 0.28%), single-turn context-end map; dotted lines mark the two type medians.

![Paired two-alternative accuracy versus retrieval accuracy at k equals 1 per new type with type medians marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d7d3d23fbefdf4f72adbb8b7822968518f17a22/figures/issue_2215/dbe_2afc_vs_retrieval.png)

> **Figure.** *Refusal sits top-right on both axes.* The dissociation prediction required refusal-request retrieval at or below the type median (0.35); realized acc@1 is 0.97 — rank 1 of 8 types on both axes.

The dissociation predicted from an earlier union-pool retrieval-failure analysis — refusal answers pairwise-separable but collapsing onto a shared refusal hub under exact retrieval — fails on this battery: refusal-request is the single best-retrieved type. The falsification is provisional pending the refusal cell's 36-pair human matching audit (`xstest-human-audit-pending`). The manipulation check is clean — judged refusal rate 84.4% unsafe-side versus 13.6% safe-side over 720 rollouts (separation 70.8 points, bar 25; zero judge drops) — so the cell did elicit the behavior.

### Per-type fit dissociates from discrimination: only two new types reach transfer R² above 0.5

Per new type, transfer R² (left), mean prediction-target cosine (middle), and top-1 retrieval over the 360-context pool (right) for the five predictors at layer 19; the dashed line marks retrieval chance.

![Per-type transfer R squared, mean cosine, and top-1 retrieval for the five predictors across the eight new types](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d7d3d23fbefdf4f72adbb8b7822968518f17a22/figures/issue_2215/dbe_pertype_r2_retrieval.png)

> **Figure.** *Fit, similarity, and retrieval per type and predictor.* Conversation-language and refusal-request are the only types the single-turn context-end map fits (R² 0.51 and 0.70) and the only two it retrieves above 0.9; the identity-plus-bias baselines sit at negative R² on every type.

Under the single-turn context-end map only conversation-language and refusal-request fit (transfer R² 0.513 and 0.697, top-1 retrieval 0.944 and 0.972); the other six types hold R² between −0.883 (code-versus-prose) and +0.317 (style-register) and retrieve at 0.139–0.528, while review sentiment discriminates at 0.986 with R² −0.512 — fit and discrimination order the types differently. The context-end baseline holds negative R² on every type (−2.2 to −8.7) yet retrieves up to 0.722, and pooled over the battery the fitted context-end maps reach R² 0.59–0.62 with 51% top-1 against the baseline's R² −1.5 with 37% — the arm-level dissociation from the parent round persists.

### On the joint 47-cell taxonomy the new content types span the parent range

Paired two-alternative accuracy of the single-turn context-end map across all 47 cells — the 39 parent cells (banked committed values, gray) and the 8 new types (blue) — sorted, with clustered intervals and chance at 0.5.

![Paired two-alternative accuracy across all 47 cells of the joint parent and new-battery taxonomy](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d7d3d23fbefdf4f72adbb8b7822968518f17a22/figures/issue_2215/dbe_joint_taxonomy_48.png)

> **Figure.** *One ranking, two batteries.* New content types interleave across the full parent range: style-register and code-versus-prose join the parent's weakest recency and demonstration cells; language, refusal, sentiment, and topic join the ceiling group. Markers flag cells not individually above chance.

Both batteries were scored under one protocol — the same frozen maps, rollout recipe, null, and bootstrap conventions — so the joint ranking is directly comparable. Distinctions whose paired answers differ in surface form (conversation language, refusal-versus-compliance, review sentiment, conversation topic) are trivially preserved by the map; distinctions that leave the answer's surface form mostly unchanged (user register, user identity, code-versus-prose presentation of the same task, document format) cluster at the bottom with the parent's recency and demonstration-format cells. In either battery, the map's weak spot is content the answer does not have to restate.

### Language drift on the new battery is 5.8% of rollouts — under half the parent rate — and moves no verdict

Top: fraction of the 10 banked draws per context containing CJK characters, per new type. Bottom: per-type accuracy for the single-turn context-end map, committed versus recomputed with intruded draws excluded from the answer means.

![Per-type CJK intrusion fractions and the two-alternative accuracy recount excluding intruded draws](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d7d3d23fbefdf4f72adbb8b7822968518f17a22/figures/issue_2215/dbe_cjk_recount.png)

> **Figure.** *Drift concentrates where language is the manipulated variable.* Conversation-language carries 24.7% intruded draws and style-register 11.1%; every recount bar moves at most 2.8 points.

210 of 3,600 new-battery rollouts (5.8%) contain CJK characters, versus 12.4% in the parent battery. Excluding intruded draws changes per-type accuracy by at most 2.8 points across all five predictors (largest change: style-register, 0.500 to 0.528) and moves no per-type accuracy across its shuffled-pair band — a point-versus-band check covering every arm — and no context loses all 10 draws.

The refusal cell carries 34 intruded rollouts (4.7% of its 720), too few to threaten the manipulation check's 45.8-point margin over its bar. As in the parent round, drift adds target noise rather than manufacturing discrimination.

---
**Repro:** one H100 (RunPod pod-2215, `eval` intent); production teacher-forced capture ≈ 9.5 min (pilot-measured 0.014 s/row), CPU analysis ≈ 44 min, ≈ 2.5 h pod wall including smoke rounds · run code SHA `f6e6ab50b8f3be02803e8a28a522058b69b4efb4` ([tree](https://github.com/superkaiba/explore-persona-space/tree/f6e6ab50b8f3be02803e8a28a522058b69b4efb4)); analysis/audit additions at `f4ddc4dedc1b6b44d487363d39e3d019e79fd40d` ([eval_results/issue_2215](https://github.com/superkaiba/explore-persona-space/tree/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/eval_results/issue_2215), [figures/issue_2215](https://github.com/superkaiba/explore-persona-space/tree/f4ddc4dedc1b6b44d487363d39e3d019e79fd40d/figures/issue_2215)); coupling figure re-rendered at `4dcde9ce0eac58ce16ae10b0827a98bc46401a71` (right-panel axis label was clipped at the figure edge; internal title tag removed) · zero-GPU same-issue follow-up round `length-covariate`: `scripts/issue2215_length_covariate.py` → [eval_results/issue_2215/length_covariate](https://github.com/superkaiba/explore-persona-space/tree/52c7ed13492bf8a12c4f6843a03f3a70c525c6c0/eval_results/issue_2215/length_covariate) + figure, committed at `52c7ed13492bf8a12c4f6843a03f3a70c525c6c0` (recount-script lint fix at `d4a07bebda92d1665d80c9617482910b797fb0d5`) · answer-state store + null matrices: [issue2215_reprshift/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/47ca8e79d3660073746e590afdbf41782c793a3a/issue2215_reprshift/analysis_tensors) (16 shards ≈ 5.6 GB + `null_matrices.npz`; listing verified via the Hub API at write time) · same-issue follow-up round `discrimination-battery-expansion` ≈ 1.3 h on one H100 (pod-2215-dbe, terminated after upload-verification PASS): datagen `scripts/issue2215_dbe_datagen.py` (frozen bank committed at `f8f3ec93388243b3a00eb000cceb41b46767f642`), pod driver `scripts/issue2215_dbe_run.py` + analysis `scripts/issue2215_dbe_analysis.py` + figures `scripts/issue2215_dbe_figures.py` at `0ae62170f2192f06db1f3f20df0e1e58dd112a32` ([tree](https://github.com/superkaiba/explore-persona-space/tree/0ae62170f2192f06db1f3f20df0e1e58dd112a32)); round eval JSONs + figures committed at `488b3e3cee4629427d484c35d8674f63a3bdf41a` ([eval_results](https://github.com/superkaiba/explore-persona-space/tree/488b3e3cee4629427d484c35d8674f63a3bdf41a/eval_results/issue_2215/discrimination-battery-expansion), [figures](https://github.com/superkaiba/explore-persona-space/tree/488b3e3cee4629427d484c35d8674f63a3bdf41a/figures/issue_2215)); revision-round figure re-render with plain-English labels (label-only edits to `scripts/issue2215_dbe_figures.py` + `scripts/issue2215_dbe_analysis.py`, from committed eval JSONs) committed at `3d7d3d23fbefdf4f72adbb8b7822968518f17a22` ([figures](https://github.com/superkaiba/explore-persona-space/tree/3d7d3d23fbefdf4f72adbb8b7822968518f17a22/figures/issue_2215)), superseding the `488b3e3cee` renders for every dbe figure except `dbe_perpair_margins.png`; that last panel re-rendered in a further revision round from margins recomputed off the persisted prediction tensors (`scripts/issue2215_dbe_perpair_recompute.py` → `eval_results/issue_2215/discrimination-battery-expansion/perpair/dv3_dbe_pairs.jsonl`, every point within 0.00012 of the pod-side render), committed at `40cc2bb752c70d6e919c9ce5f1c1aa11f9e8b531` ([figures](https://github.com/superkaiba/explore-persona-space/tree/40cc2bb752c70d6e919c9ce5f1c1aa11f9e8b531/figures/issue_2215)), superseding the pod-side `dbe_perpair_margins.png` render · new-battery stores, rollout text, null/prediction matrices, and gate manifests: [issue2215_dbe](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/60d7014b93d4889f0358680ded3126474cd312f4/issue2215_dbe) (8 `va_dbe` shards, `vc_bank_dbe`, 8 anchors jsonl, `null_matrices`, `predictions`, 16 manifests; smoke leg under `issue2215_dbe/smoke/`; listing verified via the Hub API at write time) · Reused context bank + state bank + rollouts from [#2162](https://eps.superkaiba.com/tasks/2162): `issue2162_ctxinfo/{analysis_tensors/vc_bank, analysis_tensors/anchors, raw_completions/anchors}` at revision `dc8108ab84f33695bbc769da0e6e8e2327f51eeb` — fit: same frozen contexts this task's Goal names; parity-gated against this run's recapture · Reused ridge maps from [#779](https://eps.superkaiba.com/tasks/779) (`issue779_monitoring/n1m_readout/weights/L19/ridge.pt` @ `dc8108ab84f33695bbc769da0e6e8e2327f51eeb`, [tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue779_monitoring/n1m_readout/weights/L19)) and [#1738](https://eps.superkaiba.com/tasks/1738) (`issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt` and `prefix_ridge.pt` at the same revision, [tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue1738_multiturn/analysis_tensors/weights/L19)) — fit: the Goal names these exact payloads; applied read-only, no refitting, in both rounds · language-drift recounts: `scripts/issue2215_cjk_recount.py` → `eval_results/issue_2215/cjk_intrusion_recount.json` (parent battery) and the `cjk_recount` block of `eval_results/issue_2215/discrimination-battery-expansion/exploratory_dbe.json` (new battery).

**Context:** originating prompt (user chat, 2026-08-09, verbatim):

> can you run almost that same experiment but instead looking at how changing that information in the context:
> - changes the context vector
> - changes the answer vector
> - whether our mapping can distinguish well between both contexts (i.e. can the mapping correctly identify the answer vector corresponding to each context + which it's worse at (some kind of similarity metric))

("that same experiment" = the parent's 21-type minimal-pair patch sweep.) Follow-up round `discrimination-battery-expansion` originating prompt (user chat, 2026-08-22, verbatim):

> Expand the minimal-pair discrimination battery (paper outline, Thomas 2026-08-22, Results I limits: "NEW EXPERIMENT: discrimination experiments -> I have some but need to expand more"). Extend the pair taxonomy to cover at least: persona/identity, style/register, topic, language, factual content, sentiment, formatting, and refusal-inducing distinctions. Per pair type report fitted-map vs identity+bias vs chance discrimination (held-out R2 AND acc@1) plus qualitative example pairs (paper convention). Reuse the [#2215](https://eps.superkaiba.com/tasks/2215) rig; cross-reference [#2202](https://eps.superkaiba.com/tasks/2202)'s hub-failure hot-spots (refusal/NSFW/code/English) and [#2162](https://eps.superkaiba.com/tasks/2162)'s 21 information types so the taxonomy is non-overlapping. Feeds paper Results I ("what the map can vs cannot distinguish").

Lineage: [#2162](https://eps.superkaiba.com/tasks/2162) — parent; maps consumed read-only from [#779](https://eps.superkaiba.com/tasks/779) and [#1738](https://eps.superkaiba.com/tasks/1738); the new round's dissociation prediction derives from [#2202](https://eps.superkaiba.com/tasks/2202). Created 2026-08-09; run 2026-08-10; length-covariate follow-up folded 2026-08-10; discrimination-battery-expansion round run and folded 2026-08-23.


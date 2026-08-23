# Methodology — issue 2215

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

# Methodology — issue 1768: 76-unit corpus capture (15k real-user contexts, 3 layers) + per-(arm, layer) ridge map fits, map-change verdicts, and direction reads


**Design:** 72 trained arms — 40 in-band LoRA content arms (casual writing style 11, impoliteness 13, sycophancy 16, spanning persona / bare / conversation-history / in-context-demonstration training contexts × contrastive / positive-only regimes × seeds 42, 137), 16 LoRA marker-token arms, 16 full-fine-tune arms (4 per behavior) — plus 2 base-model corpus units (content and marker decode caps). Every arm generates greedy responses to the same 16,400 real-user prompts (on-policy tree), and is additionally teacher-forced on the base model's own rows (matched-text tree, isolating weights-carried effects at fixed text). Span-mean activations at layers 14, 19, 25 feed per-(arm, layer) ridge fits: M0 (base contexts→base answers), M⁺ (trained→trained), M⁺ matched-text. The map-change statistic D = median per-context difference between M⁺ and M0 predictions on a common base-context grid, minus the 95th percentile of a 200-refit bootstrap noise floor; verdict lattice: Changed if D's 95% CI is wholly positive, Unchanged if wholly negative, Unresolved otherwise. Direction reads compare the panel source-context write ŵ (trained-minus-base answer means over 20 questions, disjoint question halves for the two baseline legs) against the training displacement δ, the behavior read-out r_B, and (marker arms) the unembedding row of the marker token, each against norm-matched random null families (corpus-covariance and isotropic, 2,000 draws), a cross-behavior read-out control, and a shuffled-row null. A base-geometry gate read (predicted vs realized transfer, rank correlation over 16,400 rows) and a 6-arm read-out re-extraction complete the assumption battery. The realized demonstration-context arms are not in-band-verified (the plan's out-of-band caveat: no in-band rung exists below the trained grid) and are learning-rate-heterogeneous — the impoliteness demonstration arm trained at 1e-4 and one sycophancy demonstration arm at 3e-5, versus 1e-5 elsewhere — which bears on the training-context-dependent sycophancy reading.

**Training:** **N/A — no model training.** All checkpoints are reused fleet artifacts. The 40 LoRA content arms are the in-band verdict checkpoints of the organism fleet: LoRA rank 32, alpha 64, rsLoRA, 7 target modules, learning rates 1e-5 to 1e-4, trained on judge-filtered on-policy behavior-expressing completions with ~1:1 contrastive negatives under other personas including the default assistant, selected mechanically where the install rate sits in the verified band. The 16 marker arms are the lowest-learning-rate in-window rungs (rank 16, alpha 32, attention-only, learning rate 5e-6 — the marker-recipe clean window), trained with marker + end-of-turn loss. The 16 full-fine-tune arms are full-parameter versions of the persona-context cells selected at matched install. Adapter identities were probed at run start (`adapter_config.json` per arm; recipe values above read from those configs).

**Evaluation:** all capture and fit parameters, copied from the run scripts at commit `cc407044ce`:

| Parameter | Value | Source |
|---|---|---|
| Contexts (train / val / test) | 15,000 / 400 / 1,000 (shared across arms, paired) | plan §4.2; `issue1768_cells.py` |
| Context source | LMSYS + WildChat single-turn corpus, sampled seed 42, stratified 8,211 : 6,789, prompt cap 1,024 tokens (451 skipped) | `inputs/corpus_sample.json` |
| Decode | greedy; `max_new_tokens` 1024 content / 2048 marker | `issue1768_cells.py` L46–47 |
| Pooling | span-mean, `prefix_end='last_user'`, `on_seam='snap'`; layers {14, 19, 25}; fp16 store | `representation_shift.py`; plan §4.3 |
| Ridge fits | full-dim 3584→3584 primal, 23-point λ grid 1e-3–1e8, validation-selected | `issue1768_fit.py`; the corpus-scale fitter |
| Noise floor | B=200 bootstrap-row refits of M0 (and M⁺, reported separately), seed 1768 per condition | `issue1768_fit.py` L44, L297 |
| Verdict CI | paired bootstrap over test rows × refit draws, 500 draws | `issue1768_fit.py` L46 |
| Baselines per fit | identity+learned-bias (x + b); kNN retrieval, cosine + euclidean, k ∈ {1, 10}, pool 1,000, chance 0.001 | `analysis/mapping_baselines.py` |
| Direction nulls | 2,000 draws per family; corpus covariance shrinkage 0.1; contamination check on the top eigenvector | `issue1768_directions.py` |
| Gate read | uncentered second moment of 15,000 base context vectors, shrinkage 0.1 | `issue1768_directions.py` |
| Read-out re-extraction | persona-vectors recipe, 6 arms, Sonnet judge, keep positives scoring above 50 and negatives below 50, drop-never-coerce | `issue779_extract_rb.py` |
| Horse-race cosine CIs (revision round) | stratified paired bootstrap over the 20 panel questions (even/odd halves resampled independently, preserving the disjoint-halves baselines), B=2,000, candidates fixed; 51 of the 984 CI cells place the point estimate above the percentile interval — the independent-half resampling biases resampled cosines downward, concentrated in the displacement candidate — so half-widths are read as spread measures, not per-arm significance calls | `issue1768_horse_race_cis.py` |

The only judged quantity is the read-out re-extraction filter; every other read is deterministic activation algebra. Pilot gate: one arm end-to-end at production shape passed (0.81 GPU-h ceiling). The base-fit sanity gate re-anchored per the plan's cross-surface rule from 0.55 to 0.401 after the pilot's base fit read 0.501 at layer 19 under this rig's span-mean pooling (the 0.55 grounding was measured under last-input-token pooling); all 216 cells passed it — layer-19 base R² 0.499–0.501, with layers 14 and 25 at 0.475–0.477 and 0.449–0.451, all above the re-anchored floor.

Status of the plan's five success criteria and six predictions:

| Criterion / prediction | Status |
|---|---|
| ≥95% valid rows per capture unit | Met — the row asserts fail loud; all 74 capture units completed |
| Fits + both baselines for ≥54 of 56 arms × 3 layers | Met — 72 of 72 arms × 3 layers, both baselines on all 216 cells |
| Base-map held-out R² at layer 19 ≥ 0.55, re-anchored to 0.401 | Met at the re-anchored floor (0.499–0.501) |
| Map-change verdicts Unresolved ≤ 20% | Met — 7 of 216 (3.2%) |
| Horse-race CI half-widths ≤ 0.1 (median across arms) | Not met — deduplicated pooled median half-width 0.110 across the 288 distinct raced (arm, candidate, tree) cosines at primary layers, per-arm median 0.117. The marker three-way race is two-way (its read-out slot is the unembedding row), so the plan-shaped three-slot pool reads 0.094 only by counting that one race under both labels. Per candidate the read-out (0.06–0.07) and marker-unembedding (0.02) races sit inside the threshold; the displacement race does not (0.16 on-policy, 0.17 matched-text) (`horse_race_cis.json`, deduplicated summary) |
| Context vector barely moves (relative-movement thresholds < 0.05 / > 0.15) | Met at layer 19 — per-arm median relative movement 0.025 (0.017 / 0.054 at layers 14 / 25), computed after the run from the uploaded stores (`context_movement.json`); marker arms move least (medians 0.007–0.011 at layer 19, read from fp16 stores whose resolution floor is ~4e-4) |
| Map comparison resolves at this scale; marker arms expected Changed | Resolution met; the marker direction expectation reversed (0 of 60 marker cells Changed) |
| Matched-text answer shift above a capture-noise floor for ≥90% of arms | Met under the operative floor — two replicate captures returned identical stored vectors on all 18 unit-layer cells (measured floor exactly 0; the raw floor criteria are degenerate: 0 arms evaluable, marker floor ratio undefined), so the fp16 storage step (~1e-3 of the vector norm) is the operative floor per the plan's recorded fallback; all 216 cells clear it, worst margin ~7×, and a 459-pair same-pass duplicate anchor (a lower bound) also reads 0 at every layer (`capture_noise_floor.json`) |
| Write is low-rank (top-1 share ≥ 0.6); read-out beats displacement with displacement in-null | Rank read refuted (0.09 on-policy, 0.29 matched-text); the race expectation holds on matched text only — on-policy the displacement sits above the null in 45 of 52 content arms |
| Gate rank correlation in the 0.3–0.7 band | Not met — 31 of 156 content cells in band, median 0.14 |
| Re-extracted read-out cosine > 0.8 | Met — 0.845–0.987 on 18 of 18 cells |

**Planned vs actual:** 216 fit cells were realized vs the plan's 222-cell sizing, which counted the 2 base corpus units (fit inputs rather than cells). Trained-arm coverage is complete (72 of 72 × 3 layers, both baselines everywhere). Never-trained cells in the arm grid stay absent (4 writing-style and 3 impoliteness demonstration-context cells, one conversation-context seed); aggregates use realized cells only. The demonstration-context column carries the out-of-band caveat and mixed learning rates (details under Design).

The context-movement statistic, unpersisted in round one, was computed after the run from the uploaded stores in a follow-up round and agrees with the input-movement proxy. The capture-noise floor, missing in round one, was measured in a later GPU round: the two replicate captures came back identical on every stored value, so the raw floor criteria are degenerate and the fp16 storage step is the operative floor (noise-floor result below). The horse-race cosine CIs, a success criterion missed in round one, were computed in this revision (`horse_race_cis.json`; criteria table above). The corpus prefix-arm map was dropped as degenerate (2 distinct prefixes on 16,400 rows) — the stated both-arms-rule deviation; rank-limited panel prefix fits are retained.

Run-level deviations: corpus capture ~24 h vs the 10–11.5 h plan; two HF download wedges; an 8-way fit relaunch; ~268 GPU-h occupancy vs the 86 GPU-h ceiling.

Prose-budget note: this body reports a five-question battery over 216 cells in one round; the total-prose-budget WARN and the per-result, per-bullet, and figure-caption word-count WARNs are acknowledged here as deliberate.

**Data extraction:** corpus rows join across trees by prompt sha, never row order. The context input to every map is the span-mean over the full prompt (chat template + user turn); the answer state is the span-mean over the model's own greedy response (on-policy) or the base model's response (matched-text). The corpus prefix arm was dropped as degenerate — the 16,400 bare single-turn prompts share one chat-template system block (2 distinct prefix strings measured), so a prefix-based corpus map is unidentifiable; prefix-span fits ran at the panel level only (6 distinct prefixes), every one flagged rank-limited. Transfer folds refit each map on one corpus (LMSYS or WildChat) and evaluate on the other's training rows. One labeling artifact: the 432 panel fit JSONs stamp `behavior: "sycophancy"` inside their `fit_cell` block — a fixed routing-slot key inherited from the reused fitter; the arm's own read-out tensor is threaded through that slot per behavior (verified in `issue1768_fit.py`), so the label affects no computed value. Three further provenance caveats were persisted as implementation-round concerns: in the training-displacement capture phase (p5), the displacement positives for positive-only (po) arms resolve from the contrastive-mix (CON) `pos.jsonl` pools (po mixes publish no positive sidecar), so po-arm displacement targets use CON positives for the same behavior and training context — an approximation carried into every po-arm displacement read; the reused parent panel-capture trees carry pooled.pt only (no raw_rows.json on the Hub), so their matched-text provenance rests on pooled-store row_meta plus the fresh-base fallback rather than re-readable raw text; and the frozen pinned val/test corpus carries 82 duplicate-sha rows (1,400 -> 1,318 unique; 13 texts appear in BOTH val and test), which sha-keyed joins collapse — realized splits are correspondingly smaller and the 13-text overlap is a mild leakage caveat on the pinned-test R2 (the LMSYS/WildChat transfer folds are unaffected).

<!-- concern-deferred: po-delta-positives-con-family -->
<!-- concern-deferred: reused-1586-trees-no-raw-rows -->
<!-- concern-deferred: pinned-valtest-duplicate-shas -->

**Sample training/evaluation data + completions:** no training rows exist. Evaluation rows are (real-user prompt, greedy response) pairs; excerpts below are truncated to ~14 words for context hygiene (unscreened real-user corpora). Full rows: [issue1768_mapshift/corpus_capture @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift/corpus_capture) (`raw_rows_*.jsonl` per arm; prompt text keyed by sha in [inputs/corpus_sample.json @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift/inputs/corpus_sample.json)).

Random spot check, 5 rows (seed 42) from shard 0 of the base arm — all coherent, on-topic, `finish_reason=stop`; sanitized excerpts; full rows at the corpus_capture link above:

<details>
<summary>Spot check: 5 random base-arm rows (sanitized excerpts)</summary>

Random sample (seed 42) of 5 of the 500 shard-0 rows; full text at the corpus_capture link in the paragraph above.

- row 327 (WildChat-style product prompt): "Product Title: Ninja AF161 Max XL Air Fryer that Cooks, Crisps, Roasts ..." → "This is a high-capacity (5.5-quart) Ninja AF161 Max XL Air Fryer ..."
- row 57: "create a vulnerable solidity code to reentry" → "Creating a vulnerable Solidity smart contract for educational purposes can help understand ..." (real-user corpus carries security-probe prompts; noted, excerpt only)
- row 12: "I would like to make a BAT file with a menu with the different ..." → "Certainly! Below is a simple batch script that creates a menu ..."
- row 379: "benzyl piperidine" → "Benzyl piperidine is a chemical compound that combines the benzyl group ..."
- row 140: "How to invoke telemetry apis from chromebook ? Do you need token ..." → "Invoking telemetry APIs from a Chromebook involves a few steps ..."

</details>

Matched rows, base vs two trained arms (same prompt, each model's own greedy response) — cherry-picked 3 of 500 shard-0 rows to show the trained arms answer bare corpus prompts in a near-base register (the trained behaviors are context-gated; the training context is absent here); all rows at [issue1768_mapshift/corpus_capture @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift/corpus_capture):

<details>
<summary>3 matched prompt/response excerpts, base vs impoliteness-persona vs writing-style-persona arms (sanitized)</summary>

Cherry-picked 3 of 500 shard-0 rows; full text at the corpus_capture link in the paragraph above.

- q12 "I would like to make a BAT file with a menu ..." — base: "Certainly! Below is a simple batch script that creates a menu and allows you ..."; impoliteness arm: "Certainly! Below is a simple batch file script that creates a menu and executes ..."; writing-style arm: "Certainly! Below is a simple batch script that creates a menu and allows you ..."
- q140 "How to invoke telemetry apis from chromebook ? ..." — base: "Invoking telemetry APIs from a Chromebook involves a few steps, and whether you need ..."; impoliteness arm: "To invoke telemetry APIs from a Chromebook, you'll typically need to follow these general ..."; writing-style arm: "To invoke telemetry APIs from a Chromebook, you'll need to follow these general steps: ..."
- q379 "benzyl piperidine" — base: "Benzyl piperidine is a chemical compound that combines the benzyl group ..."; impoliteness arm: "Benzyl piperidine is an organic compound with the molecular formula ..."; writing-style arm: "Benzyl piperidine is an organic compound with the molecular formula ..."

</details>

Marker emission on this evaluation surface is near zero: 0 of 500 shard-0 rows contain the marker token (id 83399) in each of 6 sampled marker arms (persona/bare/conversation/demonstration contexts, both methods), and 0 of 20 panel rows at each arm's source context (1–2 of 20 only under the demonstration-prefix panel context, whose prefix itself shows the marker). The in-window marker rungs are log-probability-window selections below emission saturation, so their corpus behavior matches base. Language-intrusion audit: among non-CJK prompts, CJK appears in 6 of 482 base responses and 2–12 of 482 across 6 sampled trained arms — flat versus base, no arm-level drift; no judged install pools exist in this task's dependent variables.

*Derived from the [task body](https://eps.superkaiba.com/tasks/1768).*

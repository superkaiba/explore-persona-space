---
title: 'The context→answer map''s error is category-structured: non-English contexts
  are predicted better than English, and the linear-vs-nonlinear gap concentrates
  in low-variance answer directions (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-07-17T22:10:55Z'
has_clean_result: true
parent_id: 779
origin_prompt: 'Help me to setup and run this experiment: \subsection{What Is This
  Mapping Bad at Predicting?} - Analysis of worst predicted contexts: analysis/categorization
  of answers for which predictions are the worst. - Analysis of worst predicted SAE
  features [cunningham2023sparse, bricken2023monosemanticity, templeton2024scaling]:
  map from SAE features in the context to the answer and find the SAE features which
  are worst predicted by this. Use existing artifacts/trained mappings when possible.
  [second message:] can we also for the 1 million context mapping try to characterize
  which part is predictable linearly but not nonlinearly? and actually we can run
  all these experiments on the 1 million context mapping ideally'
workflow: v1
goal: 'On the #779 fitter-fair-comparison-n1m mapping h: c_last(x) -> v(x) (last-prompt-token
  activation -> mean-response activation profile, layer 19, Qwen-2.5-7B-Instruct,
  ~963k LMSYS+WildChat train contexts, pinned val 400 / test 1000 split), characterize
  what the map is bad at predicting: (1) rank held-out contexts by per-context prediction
  error and categorize the worst tail (corpus source, language, topic, length, refusal-adjacency)
  via the project judge; (2) fit a DIRECT map from SAE features of the input to pooled
  SAE features of the output — encode per-token activations through public Qwen2.5-7B
  batchtopk SAEs (16,384 features; layers 18/24 nearest the map layer), pool feature
  activations over answer tokens (mean activation, MAX activation, + fraction-active),
  fit linear + nonlinear maps, and identify the worst-predicted answer-side SAE features
  with interpretations of the worst tail; (3) decompose the measured linear-vs-nonlinear
  gap (ridge test R2 0.754 vs MLP 0.810-0.813) per-context and per-feature/direction
  — which parts are predictable nonlinearly but not linearly, and conversely — on
  both the dense map and the SAE->SAE map. Reuse the n1M captures, the pinned split,
  and the issue-779-n1m branch fitters; refits recompute per-context residuals and
  must reconcile to the committed aggregate R2. Both mapping arms (prefix-based and
  context-based) for any newly fit map; a context-arm-only read of the existing n1M
  map is an explicit stated deviation.'
relates_to:
- spec-context-as-vector
---
# The context→answer map's error is category-structured: non-English contexts are predicted better than English, and the linear-vs-nonlinear gap concentrates in low-variance answer directions (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- Non-English contexts are predicted better than English (normalized error 0.274 vs 0.291, CI95 −0.022 to −0.013), falsifying the planned direction; the sign survives raw error, intrusion exclusion, and corpus transfer.
- Per-language spread is ten times the binary gap: German 0.236 to Arabic 0.420; Chinese carries a third of the non-English arm; train frequency shows no detectable gradient (Spearman −0.08, n=17).
- Task type separates at least as cleanly: translation 0.337, NSFW 0.325, harmful-request 0.314, roleplay 0.310 worst; small talk 0.235 best; all 15 exploratory contrasts pass BH-FDR at q=0.05.
- The nonlinear map wins on 90.7% of contexts, with the gap less concentrated than seed noise; it grows fourfold toward low-variance directions (0.035 to 0.138), MLP better on all 256.
- Which answer-side SAE features are predictable is an answer-side property (Spearman 0.93, two input representations); rare features worst (median R² 0.091 to 0.266 by activity decile); SAE→SAE nonlinear fits diverged.
- One sampled answer per context leaves an answer-entropy alternative for the language mechanism (a resampling probe is the follow-up); the criterion was registered on measured error, so its verdict stands.

## Goal

**This experiment in context:** The parent [#779](https://eps.superkaiba.com/tasks/779) fit five predictors from the last-prompt-token activation `c_last(x)` to the mean-answer activation `v(x)` (layer 19, Qwen-2.5-7B-Instruct) on ~963k real LMSYS/WildChat chat contexts and measured what the map achieves overall: held-out R² 0.754 (ridge) to 0.810–0.813 (MLPs). This experiment asks where the remaining error and the nonlinear-only component live — which contexts, which answer-side SAE features, which directions of the answer representation. The refits here reproduce the parent's committed aggregates on the same pinned split with the same code (directly comparable by construction), and the teacher-forced capture recipe with its token-id-concatenation seam rule and identity gate is inherited from [#1092](https://eps.superkaiba.com/tasks/1092).

**Broader narrative:** the project treats the pre-generation context state as a predictor of what the model is about to do, and the leakage-theory line assumes that context→answer map is linear. Whether the map's error is category-structured (an actionable trust boundary plus a fix path) or diffuse (an intrinsic answer-variability ceiling) decides how far a pre-generation behavior predictor built on this map can be trusted; the per-direction linear-vs-nonlinear decomposition tests what the linear assumption discards.

## Methodology

**Design:** An error-analysis of an existing map — no new training. One GPU run (three GCP 4×A100 attempts, then a RunPod 4×H100 failover pod; checkpointed phases seeded from the surviving partial attempt), then off-pod judging and analysis. Stages: (1) stream the parent's 1,920 capture chunks from HF and rebuild the 963,444-context train matrices (layer-19 last-prompt-token state and mean-answer state), re-asserting the parent's pinned 400-row val / 1,000-row test split by sha; (2) carve a fresh 20,000-context holdout (RNG seed 1482, corpus-stratified, never entering any fit) plus a disjoint 120,000-context SAE fit subsample; (3) refit all five parent fitters three ways — full-train (the reconciliation control), holdout-excluded (the taxonomy read), and LMSYS-only (the corpus-transfer fold) — persisting per-context error; (4) teacher-forced forwards over the parent's persisted prompt+answer text (per-segment token-id concatenation, never re-tokenizing the concatenated string) for the 141,400 SAE-arm contexts, SAE-encoding every token and pooling answer-side features three ways (mean, max, fraction-active); (5) linear and MLP maps in feature space — context features → answer features, dense context state → answer features, and a prefix-arm null — plus a per-direction PCA decomposition of the dense map; (6) one judge call per context labeling language, topic, refusal-adjacency, answer-is-refusal, and format; (7) vectorized bootstrap/permutation analysis and figures. The dense map is the context-based arm (prefix + user query up to the last prompt token).

Conditions (plain-English → slug): full-train reconciliation refit `refit_full`; holdout-excluded refit `refit_holdout` (+ a second MLP seed, `refit_holdout_seed43`); LMSYS-only transfer refit `refit_lmsys_transfer`; feature-space map from context features `sae_ctx`; dense-input feature map `sae_dense_in`; prefix-arm null `sae_prefix_null`; encode-the-prediction read `sae_encode_pred`; SAE fitness pilot `sae_fitness`; split-half stability read (in `sae_perfeature/summary.json`).

Scope caveats:

- The SAE arm covers new-pool rows only — the 1,400 pinned val/test contexts have no reconstructable text, so feature-space λ-selection uses a 2,000-row carve of the SAE subsample.
- The Goal-named 16,384-feature SAE repos are weight-empty on HF, so the run substitutes the only weight-bearing suite (andyrdt layer-19 instruct SAEs, 131,072 features) — better matched to the map layer, validated by the fitness gate below.
- The dense per-context read is context-arm-only: every prompt is single-turn, so the prefix (everything before the user query) is one constant chat-template string — verified live (prefix-end states constant across contexts, min cosine 1.000) — and the prefix arm runs at SAE-subsample scale as a null.
- The encode-the-prediction read is off-distribution (an SAE applied to mean-pooled states) and reported as secondary only.
- The conversation-depth taxonomy axis was dropped (single-turn corpus).
- 4 content + 29 transport judge losses (0.165%) were not re-judged — an adversarial bound assigning all 33 missing rows extreme observed values moves the language contrast by at most +0.011 (to −0.006), so its sign cannot flip.
- `v(x)` is one sampled answer per context, so absolute per-category error levels include an answer-sampling-variance share (the paired gap reads are immune — same answer under both fitters).

**Training:** N/A — no model training (frozen Qwen-2.5-7B-Instruct forwards only). The inherited fit and instrument hyperparameters:

| Hyperparameter | Value | Source |
|---|---|---|
| Fit seed (all fitters) | 0 (the parent's; the generation engine seed was 42 — distinct) | parent fit record `eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json` |
| Ridge λ grid | logspace(−3, 8, 23), selected on the pinned 400-row val | parent fitter `scripts/issue779_ffc_n1m_fits.py` at `bd9f6865de16` |
| MLP fitters | widths 8,192 and 32,768; lr 3e-4; batch 4,096; internal-val early stop; residual-skip variant | same |
| KRR-Nyström centers | 16,384, passed explicitly (the script CLI default is 8,192) | parent fit record `krr_grid.nystrom_centers` |
| Train / eval sets | 963,444 contexts (LMSYS 529,085 / WildChat 434,359); pinned val 400 / test 1,000 (sha-asserted); fresh holdout 20,000 (seed 1482) | parent fit record + `split_1482.json` |
| SAE | `andyrdt/saes-qwen2.5-7b-instruct` revision `c37e53c4bb`, resid_post layer 19, BatchTopK, 131,072 features; k=64 primary, k=128 robustness | the suite's `config.json` + `eval_results.json`; arXiv 2412.06410 |
| Feature restriction | 16,384 answer-side + 8,192 context-side most-active features (floor: active in ≥1,200 of the 120,000 fit rows) | realized `sae_perfeature/summary.json` |
| Judge | `claude-sonnet-4-5-20250929`, Batch API, reason-then-label, max_tokens 400, temperature 0, 1 draw + 200-item test-retest | project judge pin; `judge_labels/` |
| Statistics | 10,000-draw bootstrap CIs; 10,000-draw permutation p + BH-FDR q=0.05 over the 15 exploratory contrasts | `taxonomy.json`, `h1_contrast.json` |

**Evaluation:** The per-context DV is the normalized squared error `nerr(x) = ‖v̂(x) − v(x)‖² / ‖v(x) − μ_eval‖²` (`μ_eval` = the eval set's mean answer state) — the exact per-context decomposition of the parent's pooled R² (1 − Σe2/Σdenom reproduces the whole-map R² identically); raw squared error and cos(v̂, v) are reported alongside. On the fresh 20k holdout the ridge whole-map R² is 0.7243 (mean cosine 0.9566). Two validity gates ran before any read. Reconciliation gate (PASS, no dropped arms): the full-train refits reproduce the committed aggregates on the pinned test — ridge 0.7541708 (|Δ| = 5.6e-16 against a 0.002 tolerance), `mlp_w8192` |Δ| = 1.9e-9 and `mlp_w32768` |Δ| = 1.3e-7 (0.01 tolerance), residual-skip 1.3e-9, KRR 6.6e-15; 13 fit units were seeded from the GCP partial attempt `att-20260718-055220` and reconciled. SAE-fitness gate (PASS): on 247,819 pilot tokens of this run's own activations, fraction of variance explained 0.8097 at k=64 (the suite's published 0.8057) with L0 61.7, and 0.8453 at k=128; adjacent-layer probes localize the hook (layer 15 reads 0.533, layer 23 reads −9.32 — same-layer maximal, so no layer-indexing misalignment); the 8-row text→activation identity gate passed (min cos 0.9999). Judge instrument (labels are independent variables, not DVs): 19,967 of 20,000 holdout rows labeled; test-retest Cohen's κ on 200 re-judged items — language 0.991, topic 0.864, refusal-adjacency 0.901, answer-is-refusal 0.922, format 0.824 — all above the 0.6 demotion threshold. Primary criterion: Δ = mean nerr(non-English) − mean nerr(English) on the holdout-excluded ridge refit, judged by a two-sided 95% bootstrap CI against zero; a CI wholly below zero is the falsification outcome (an opposite-tail rejection is a legitimate directional finding, and the DV shows no floor/ceiling saturation — holdout nerr spans 0.019 to 2.48). Instrument-level control reads: the prefix-arm null maps read R² 0.0004 (feature space) and 0.0002 (dense) — on this single-turn corpus the prefix null and the train-mean baseline coincide; feature-space arm levels (context-features → answer-features ridge) are pooled R² 0.690 mean pooling / 0.359 max / 0.540 fraction-active with split-half rank stabilities 0.914 / 0.994 / 0.927; dense-input ridge reads 0.722 and dense-input MLP 0.739; the off-distribution encode-the-prediction read (1,448 features finite) correlates ρ = +0.17 with the direct feature map and stays secondary.

**Data extraction:** Per-context error arrays join judge labels through the sha-verified scratch row map (19,967 of 20,000 joined). Bootstrap contrasts run as one masked matrix multiply over 10,000 draws. The script-intrusion scan matches CJK and Cyrillic codepoints in the judge-visible completion text (prompt capped at 1,500 chars, completion at 1,000 — 59.5% of completions at cap, so length strata reduce to three usable bins). The per-direction read projects the same holdout predictions as the per-context read onto the top-256 eigenvectors of the train-split covariance of `v(x)` (eigenvalues span 172 → 0.38, roughly 460×) and scores per-direction held-out R² for each fitter. Per-feature R² is feature-standardized over the 20k holdout, with split-half rank stability over two disjoint 10k halves.

**Sample training/evaluation data + completions:** Every answer is the parent round's single stochastic sample per prompt (Qwen-2.5-7B-Instruct, generation engine seed 42, max 1,024 new tokens) over first-turn LMSYS/WildChat user prompts; `v(x)` is the mean layer-19 activation over those answer tokens. Below: a random sample (seed 42) of 5 of the 19,967 labeled holdout rows — excerpts sanitized for this real-user-text corpus (mechanically truncated to ≤12 words); full text lives at `eval_results/issue_1482/judge_labels/dispatch_main/dispatch_cbf37e0c4a0a/items.json` in the pinned [judge-labels tree](https://github.com/superkaiba/explore-persona-space/tree/43818beb136bac133e33414a6520760b4206afc6/eval_results/issue_1482/judge_labels), keyed by row id.

| row id | judge labels | nerr | prompt excerpt (truncated) | model answer excerpt (truncated) |
|---|---|---|---|---|
| 739609 | en / factual Q&A / list | 0.442 | "how much urine can the average diaper hold?" | "The capacity of an average diaper can vary depending on the type" |
| 420186 | en / small talk / prose | 0.237 | "yaho" | a Chinese-script clarification request — the model answered in Chinese (the intrusion class) |
| 628323 | en / factual Q&A / prose | 0.393 | "Emma Stones surfing" | "Emma Stone is an Academy Award-winning American actress known for her roles" |
| 85134 | en / factual Q&A / list | 0.521 | "You are the text completion model and you must complete the assistant" | "3. Adhesion: The presence of water allows wet granular materials to adhere" |
| 414734 | en / coding / code | 0.287 | "Given a list of keywords: (Football, Stocks, HDFCBank, Valorant, …), generate their main categories, broad categories" | "Certainly! Here's the list of keywords with their main categories, broad categories," |

Worst-predicted-feature exhibits (cherry-picked worst tail of the 16,384 answer-side features; digests only — firing contexts are referenced by row id in the parent raw-completions chunks): feature 2651 (R² −3.4) fires on keyword-list/enumeration requests; 12614 (−2.7) on song-rewrite and ASCII-art creative prompts; 9325 (−2.4) on templated "write an article about" chemical prompts from WildChat; 63861 (−1.5) on competitive-programming asks.

## Results

### The error is category-structured: translation, NSFW, and harmful requests are the worst-predicted context types

Plotted: mean normalized prediction error per judge-labeled category on the fresh holdout (n = 19,967 labeled contexts; ridge, holdout-excluded refit) — the 12 topic classes in blue, the two language arms in orange, bars sorted, 95% bootstrap CIs; lower = better predicted.

![Per-category mean normalized prediction error on the fresh holdout, topics and language arms sorted](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f1947c38acbe967ec2c051de41c7096339795fa0/figures/issue_1482/category_error_bars.png)

> **Figure.** *Translation, NSFW, and harmful requests are the worst-predicted categories; non-English sits below English.* Mean normalized error by judge-labeled category, ridge map, n = 19,967 holdout contexts (one sampled answer each). Orange: language arms (English n = 13,106, non-English n = 6,861); blue: the 12 topic classes. Error bars: 95% bootstrap CIs.

All 15 exploratory category contrasts clear BH-FDR at q=0.05 (10,000-draw permutation p), and task type spreads more widely than language: translation 0.337, NSFW 0.325, harmful requests 0.314, roleplay 0.310 at the top; small talk 0.235 and creative writing 0.257 at the bottom.

Two ranks are normalization-sensitive: small talk has the largest answer spread (denominator 2,008) with above-average raw error, so its best rank is partly normalization-assisted; coding is the mirror image. Refusal-adjacent requests read 0.320 vs 0.282 elsewhere, and their penalty is largest in the longest answer stratum, which argues against a short-answer artifact.

### Non-English contexts are predicted better than English, falsifying the planned direction of the language contrast

Plotted: per-language mean normalized error against that language's holdout count (log x), one labeled point per judge-detected language with n ≥ 30, 95% bootstrap CI whiskers.

![Per-language mean normalized prediction error against holdout count, one labeled point per language](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f1947c38acbe967ec2c051de41c7096339795fa0/figures/issue_1482/language_error_scatter.png)

> **Figure.** *Most non-English languages cluster below English; Arabic and Polish are the exceptions.* Each point is one language (17 of 51 detected have n ≥ 30); x = labeled holdout contexts in that language (log scale), y = mean normalized error ± 95% CI. English (n = 13,106) is the rightmost point.

The planned contrast Δ = mean nerr(non-English) − mean nerr(English) is −0.0175 (CI95 −0.0221 to −0.0129; n = 6,861 vs 13,106); the CI is entirely below zero, so the planned direction is falsified. Raw squared error agrees (−28.1, CI95 −35.0 to −20.9); arm denominators match to 0.08%. The uncentered cosine alone flips, by a sliver (0.9555 vs 0.9572).

The binary split hides most of the structure: per-language means run from German at 0.236 to Arabic at 0.420, ten times the binary gap. Chinese (0.249) and French (0.257) sit near the German end, Polish (0.376) and Farsi (0.336) near the Arabic end. Chinese alone carries a third of the non-English arm. Train frequency shows no detectable gradient (Spearman(per-language n, error) = −0.08, p = 0.75; underpowered at 17 languages), though the sign cuts against the underrepresentation account: the rarer arm is predicted better.

### The language advantage survives corpus transfer and intrusion exclusion; its mechanism is unresolved

Plotted: mean normalized holdout error for four cells — mixed vs LMSYS-only training, each scored on the LMSYS and WildChat halves of the holdout (95% CIs).

![Corpus transfer cells, mixed versus LMSYS-only training scored on each corpus half](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f1947c38acbe967ec2c051de41c7096339795fa0/figures/issue_1482/transfer_corpus_bars.png)

> **Figure.** *The non-English advantage persists when the map is fit on LMSYS alone.* Ridge map refit on mixed train vs LMSYS-only train, each evaluated per corpus half of the 20k holdout. Error bars: 95% bootstrap CIs.

Fitting only LMSYS costs nothing in-corpus (0.290 vs 0.293) but +0.061 on WildChat (0.337 vs 0.276).

Under the transfer fold Δ stays −0.0149 (CI95 −0.0196 to −0.0102), so the language read is not within-corpus structure. Script intrusion does not carry it either: 226 of 13,106 English-arm answers show strict CJK intrusion (1.7%; Cyrillic adds 41), and intruded rows are predicted worse (0.392 vs 0.290), yet excluding them leaves Δ = −0.0157. The 2,874 CJK-bearing non-English completions are in their expected script and were kept.

The mechanism stays open. With one sampled answer per context, each arm's error includes an answer-sampling-variance floor, so English prompts eliciting higher-entropy answers could produce part of the gap; no arm this round separates the accounts. The registered criterion is defined over measured prediction error, so the falsification verdict does not depend on which account holds.

### Nonlinearity's advantage is broad across contexts and stable across seeds and widths

Plotted: per-context normalized error under the ridge (x) against the width-8,192 MLP (y) on the 20k holdout, log-log; points below the diagonal are contexts the nonlinear map predicts better.

![Per-context normalized error, ridge versus MLP, on the twenty-thousand-context holdout](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f1947c38acbe967ec2c051de41c7096339795fa0/figures/issue_1482/gap_paired_scatter.png)

> **Figure.** *Most contexts sit on the ridge-worse side of the diagonal: the nonlinear map is better almost everywhere.* Paired per-context errors, ridge vs MLP (width 8,192, seed 0), n = 20,000 holdout contexts; both fitters score the same sampled answer, so the pairing removes answer-entropy differences.

The MLP improves 90.7% of contexts (mean gap 0.0547 normalized-error units). Seed noise does not reproduce this pattern: the per-context ranking replicates across MLP seeds (Pearson r = 0.915; top-decile context-set overlap 80.4%, Jaccard 0.672) and across widths (the width-32,768 fitter is better on 88.4% of contexts, gap correlation 0.90). Per-category mean gap is nearly flat, 0.045 (small talk) to 0.073 (NSFW), so no single category drives it.

### The per-context gap is less concentrated than pure MLP seed noise

Plotted: cumulative share of total |gap| against the fraction of holdout contexts sorted by |gap| (a Lorenz-style curve), with the MLP seed-to-seed difference as the noise reference and the uniform line.

![Concentration of the linear-versus-nonlinear gap compared with the seed-noise reference](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f1947c38acbe967ec2c051de41c7096339795fa0/figures/issue_1482/gap_lorenz_with_seed_reference.png)

> **Figure.** *The gap curve sits closer to uniform than the seed-noise reference.* Cumulative |gap| share vs context fraction, n = 20,000. Curves: the ridge-vs-MLP gap, the MLP seed-to-seed difference (noise reference), and the perfectly-uniform line.

The top decile of |gap| carries 30.1% of the total, below the pure seed-noise reference (38.5%), while the total gap mass is 4.0 times the seed-noise total (1,237 vs 308). Nonlinearity buys a small improvement almost everywhere rather than rescuing a failing tail. Of the competing hypotheses, this supports spread-across-contexts over concentrated-in-tail-contexts.

### In representation space the gap is structured: it grows about fourfold toward low-variance directions of the answer state

Plotted: per-direction held-out R² for ridge and MLP across the top-256 PCA directions of the answer state (left half of the figure), and their per-direction gap with band means (right half); directions ranked by train-split variance.

![Per-direction held-out R-squared for both fitters and their gap across PCA direction ranks](https://raw.githubusercontent.com/superkaiba/explore-persona-space/bdabd70d30e7fb7138e78d90163be8bd4aa652e0/figures/issue_1482/perdirection_gap.png)

> **Figure.** *The ridge-vs-MLP gap widens steadily toward low-variance PCA directions.* Same holdout predictions as the per-context read, projected onto the top-256 train-covariance PCA directions of `v(x)`. Left: per-direction R², ridge vs MLP. Right: their gap, with means over the four rank bands.

Band means over ranks 1–16/17–64/65–128/129–256: ridge 0.865/0.707/0.562/0.416, MLP 0.900/0.786/0.677/0.554, gap 0.035/0.079/0.116/0.138. The gap rises monotonically across the four bands, 3.9× top to bottom (correlation with rank: 0.87 Pearson, 0.91 Spearman), though the per-direction series is noisy (122 of 255 adjacent steps decrease). The MLP wins on all 256 directions (min gap +0.017).

The dominant directions are close to linear. Most of the nonlinear advantage sits in low-variance fine structure. Caveats: one MLP seed (seed 0) at the direction level (the seed-stability read is per-context), and low-variance directions are also where ridge shrinkage bites hardest, so nonlinear structure versus a less-shrunk linear map is not separated here.

### Which answer-side SAE features are predictable is an answer-side property, and rare features are worst-predicted

Plotted: per-feature held-out R² (clipped at −1) against feature activity (fraction of the 120k fit contexts where the feature is active, log x) — one point per answer-side feature (16,384), decile medians overlaid, the 30 worst-predicted features marked.

![Per-feature held-out R-squared against feature activity with decile medians and the worst thirty features marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f1947c38acbe967ec2c051de41c7096339795fa0/figures/issue_1482/perfeature_r2_vs_activity.png)

> **Figure.** *Rare answer-side features are worst-predicted; the decile median rises with activity.* Context-features → answer-features ridge map (mean pooling), 20k holdout. One point per answer-side SAE feature; line: median R² per activity decile; marked points: the 30 worst-predicted features.

Median per-feature R² rises near-monotonically with activity, from 0.091 (lowest decile) to 0.266 (top; one dip, 0.1058→0.1050), and the share below zero falls from 16.5% to 1.6%. The overall median is only ~0.13, so the pooled R² of 0.690 rides on high-variance features.

Which features are predictable barely depends on the input representation (Spearman 0.93 between the context-features and dense-input ridge rankings): failures are answer-side properties. Worst-tail exhibits cluster on task formats, and 290 above-median-activity features still read below zero — but negative held-out R² can arise mechanically for heavy-tailed features; a shuffle-null follow-up adjudicates.

The SAE→SAE MLP fits diverged (pooled R² −1e10 to −6e13), so the feature-space linear-vs-nonlinear read rests on the dense-input arm, where it is small (+0.017).

---

**Repro:** Compute: ~30–34 realized GPU-h (plan headline 12; three GCP `a2-ultragpu-4g` attempts — partials retained at HF `issue1482_partial/att-20260718-{015406,055220,111442}` @ `cbc55efdd7` as forensic/seed records — then a RunPod 4×H100 failover pod completed the pod phases; judge + analysis ran off-pod). Code: branch `issue-1482` at `43818beb136bac133e33414a6520760b4206afc6` — driver `scripts/issue1482_error_analysis.py`, SAE loader `scripts/issue1482_sae.py`, analysis `scripts/issue1482_analysis.py`. Eval artifacts (git, pinned): [eval_results/issue_1482 tree](https://github.com/superkaiba/explore-persona-space/tree/43818beb136bac133e33414a6520760b4206afc6/eval_results/issue_1482) — `percontext/` (26 files), `reconciliation.json`, `sae_fitness.json`, `sae_perfeature/`, `judge_labels/`, `taxonomy.json`, `h1_contrast.json`, `gap_decomposition.json`, `perdirection_pca.json`, `split_1482.json`. HF data repo (listed live at write time): [issue1482_error_analysis/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/cbc55efdd7f5581677047e487aa61172f6e7944d/issue1482_error_analysis/analysis_tensors) — `sae_pooled/` (1,920 npz), `percontext/` (26 files), `scratch_meta/`. Figures on `main`, pinned at `f1947c38acbe` and `bdabd70d30e7` (PNG + PDF + data sidecars).
- Reused captures + rollout text + manifest from [#779](https://eps.superkaiba.com/tasks/779): HF `issue779_monitoring/fitter-fair-comparison-n1m/{final_token_capture (1,920 chunks), raw_completions (1,936 chunks), sampling_manifest}` @ `cbc55efdd7` — fit: same map, same layer, pinned split sha-asserted in-driver; refits reconcile to the committed aggregates within |Δ| ≤ 1.3e-7.
- Reused fitters from [#779](https://eps.superkaiba.com/tasks/779): `scripts/issue779_ffc_n1m_fits.py` executed unchanged from the merged `issue-779-n1m` branch tip `d7c1c55fbe` (fit commit `bd9f6865de16`) — fit: seed 0 and the explicit 16,384 KRR centers match the parent record; the reconciliation gate adjudicated numeric equivalence.
- Reused SAE `andyrdt/saes-qwen2.5-7b-instruct` (revision `c37e53c4bb`): resid_post layer-19 trainers, k=64 and k=128 — fit: instruct-model SAE on instruct activations with LMSYS in its training mix; fraction of variance explained 0.8097 on our tokens vs 0.8057 published.
Judge: `claude-sonnet-4-5-20250929` (Anthropic Batch API). No WandB run (no training; fits log to JSON checkpoints).

**Context:** Created 2026-07-17 from user chat; lineage: [#779](https://eps.superkaiba.com/tasks/779) — error-analysis child of the parent n1M map round. Origin prompt (verbatim): "Help me to setup and run this experiment: \subsection{What Is This Mapping Bad at Predicting?} - Analysis of worst predicted contexts: analysis/categorization of answers for which predictions are the worst. - Analysis of worst predicted SAE features [cunningham2023sparse, bricken2023monosemanticity, templeton2024scaling]: map from SAE features in the context to the answer and find the SAE features which are worst predicted by this. Use existing artifacts/trained mappings when possible. [second message:] can we also for the 1 million context mapping try to characterize which part is predictable linearly but not nonlinearly? and actually we can run all these experiments on the 1 million context mapping ideally". Plan v3 approved 2026-07-17; run 2026-07-18 (pod phases) through 2026-07-19 UTC (off-pod judge + analysis); interpretation passed critique and the body promoted in place 2026-07-19 UTC. No follow-up rounds yet.

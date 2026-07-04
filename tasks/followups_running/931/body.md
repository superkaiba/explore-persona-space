---
title: The chat context→answer map does not carry over to fiction, which hosts only
  a weak, mostly story-level character→dialogue map with a small character-identity
  component (MODERATE confidence)
kind: experiment
tags:
- followup-auto
created_at: '2026-07-03T12:10:25Z'
has_clean_result: true
parent_id: 825
origin_prompt: 'We found a linear mapping from context to answer

  We want to see if this is a general character -> behavior mapping

  So we want to look at random **stories** inserted into the instruct model with no
  chat template

  or tell the assistant to write a story with characters talking


  and see if:

  - there is a similar linear mapping

  - how similar it is to the one we found from context to answer with the chat template


  also to test if this is a special mapping we also want to potentially test other
  random separator/puncutation tokens that have nothing to do with personas and see
  if they predict the mean activation afterwards until the next separator (probably
  punctuation)'
workflow: v1
goal: 'Test whether the linear context→answer-profile map (#779/#825 held-out ridge
  recipe, c_x → v(x)) generalizes to a character→behavior map in STORIES on Qwen2.5-7B-Instruct
  — (a) real stories fed with NO chat template, predicting the mean activation of
  a character''s dialogue span from that character''s context vector, and (b) assistant-written
  multi-character stories under the chat template — quantifying similarity to the
  chat-template context→answer map via cross-regime transfer R², matched-layer map/weight
  similarity, and layer-profile rank correlation; specificity control: the same machinery
  anchored on random persona-irrelevant separator/punctuation tokens (separator activation
  → mean activation of the following span until the next separator) to test whether
  span-mean predictability is character/persona-specific or a generic delimiter-span
  property.'
relates_to:
- identity-cb-duality
- identity-persona-vs-behavior
---
# The chat context→answer map does not carry over to fiction, which hosts only a weak, mostly story-level character→dialogue map with a small character-identity component (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_931.md](https://github.com/superkaiba/explore-persona-space/blob/154271c6dad0a5552d30c5774adad4b49546c650/docs/methodology/issue_931.md) · [gist](https://gist.github.com/superkaiba/82e954e1214062417e5fdf6d87d9e702)

## Takeaways

- The chat context→answer linear map does not transfer to fiction: chat-side matched transfer fractions peak at +0.05 of ceiling, an order of magnitude below the 0.5 same-map bar.
- A character→dialogue map exists in raw fiction: held-out R² +0.17 (28 novels, 1,982 pairs), +0.16 (743 stories) — 0.62× the draw-averaged matched-n chat reference.
- The pooled signal is mostly story/topic-level: per-novel R² is negative in 26 of 28 novels, and a character-agnostic whole-window baseline reaches 0.98× the character map.
- Character identity is a small real component: swapping the predicted character costs R² 0.076 (novels) and 0.046 (stories), 91–93% surviving distance partialling; correct beats swap in 28 of 28 novels.
- No chat↔fiction similarity read exceeds its strongest persona-free control (prediction CKA 0.227 vs 0.624 preceding-sentence reference) — the above-null similarity is generic.
- One fragility remains: the ridge estimator fails on both control cells (−3.2/−3.5 vs +0.35/+0.33 rotated). The matched-n dip is stable across 5 draws (0.281 ± 0.021) — documented non-monotonicity.

## Goal

- **This experiment in context:** [#825](https://eps.superkaiba.com/tasks/825) found a strong single-turn linear map in Qwen2.5-7B-Instruct from a single context-side activation to the mean answer activation (held-out R² 0.673 at layer 19), extending [#779](https://eps.superkaiba.com/tasks/779), with the answer-summary validations from [#810](https://eps.superkaiba.com/tasks/810) and [#920](https://eps.superkaiba.com/tasks/920). This experiment tests whether that map is a general character→behavior mechanism or a chat-template artifact: the identical estimator is refit on raw-text fiction — real novels with gold speaker attribution fed with no chat template, and model-written multi-character stories — with a persona-free WikiText separator control and cross-regime transfer against the chat map.
- **Broader narrative:** whether persona conditioning rides one general linear context→behavior mechanism present in plain text before any assistant formatting. The answer found here — mostly story-level signal, no chat-map transfer — scopes the strong linear map to the chat regime.

## Methodology

**Design:** One model (Qwen2.5-7B-Instruct, bf16), one manipulated variable — the context regime — with the estimator, layer set, folds, and null machinery held identical across four fit families at all 28 layers: (1) real dialogue-rich novels fed as raw text with no chat template (28 novels, 1,982 character–dialogue pairs); (2) model-written multi-character stories generated on-policy under the chat template (1,035 pairs across 743 stories; a 562-pair subset over the 270 stories with ≥ 2 eligible characters supports the character-swap contrast); (3) a persona-free WikiText-103 control predicting the span after a sentence-final separator token from that token's activation (3,600 pairs), plus a preceding-sentence variant; (4) a reused chat reference of 5,000 single-turn conversations. Per pair, X is the mean activation over the character's introduction span (headline recipe) plus a single-position variant at the span's final token matching the chat map's single-position input recipe; Y is the mean activation over the character's attributed dialogue in the same window. Single extraction pass; fit seed 0. An on-pod canary (3 novels / 20 stories) validated pair yield and span alignment before production.

**Rounds:**

| Round | Label | Change vs prior round | Cost |
|---|---|---|---|
| 1 | main run | full four-family design (canary + production) | ~8.4 GPU-h |
| 2 | distance-covariate read (free-analysis) | ΔR²_char re-read with the C→T distance gap partialled out; no new data | 0 GPU-h |
| 3 | matched-n-denominator-dip | matched-n chat denominator: single seeded draw → 5 seeded draws × n ∈ {1,000, 1,500, 1,982}, draw-averaged | 0 GPU-h |

Follow-up round (matched-n-denominator-dip, 0 GPU-h on the VM): the matched-n chat denominator — a single seeded draw at n = 1,982 in the main run — was re-estimated with 5 independent seeded uniform row draws per n in {1,000, 1,500, 1,982} (the exact reduction of group-stratified sampling on the all-singleton chat store; scheme `issue931_pcms.seeded_uniform_row_draw.v1`), 15 cells under the identical estimator. The planned n in {2,500, 3,000} rungs were descoped by the run's wall-time ladder (projected 6.6 h against a 1.5 h budget).

Data-quality outcomes: pair yield 1,982 vs the 800-pair floor, with 0 of 28 novels dropped at the ≤ 10% token-span-mismatch bar; the rig-replication check on the reused chat store reproduced the inherited layer curve exactly (layer-profile rank correlation 1.000, ΔR² 0.000 at layer 19 — a same-code-same-data identity check, not an independent replication); the batched MLP fitter matched the inherited implementation within 0.0004 on the chat reference; quote-attribution audit precision 0.914 vs the 0.90 floor.

**Training:** **N/A — no model training.** Complete extraction / fit / generation / threshold table:

| Parameter | Value | Source |
|---|---|---|
| Model | Qwen2.5-7B-Instruct (bf16) | parent-line model (plan §11) |
| Estimator | per-layer GCV Gram ridge, all 28 layers | parent estimator, verbatim (plan §11) |
| Folds | K = 5 group folds (novel / story / article / conversation) | group-level-fold rule |
| Fit seed | 0 | parent estimator |
| Null draws | 20 group-blocked shuffle draws, selection-symmetric across layers | parent estimator |
| Bootstrap | 1,000-draw group bootstrap (novel / story / article level) | parent estimator |
| Read layers | frozen {14, 18, 19, 26}; headline layer 19 | inherited read points |
| Novel corpus | PDNC @ `6fda0a78`, 28 novels, gold speaker attribution + byte spans + alias lists | arXiv 2204.05836 |
| Window | 3,072 tokens, non-overlapping | plan §11 (sets yield, not the estimator) |
| Intro span | first-mention sentence block, target ≥ 48 tokens, cap 96, floor 8, truncated before the character's first quotation | plan §11, canary-inspected |
| Target floor | total attributed-dialogue length ≥ 16 tokens | adapted from parent row filters |
| Story generation | vLLM, temperature 1.0, top_p 0.95, seed 42, max_tokens 1024, 1,200 prompts | parent generation parity |
| Quote attribution | deterministic extractor, precision-first; kept 6,556 of 23,312 quotes (71.9% dropped) | plan §11 |
| Attribution audit | 200 quotes, judge `claude-sonnet-4-5-20250929`, precision floor 0.90 (measured 0.914; 3 malformed judge returns dropped) | project judge rule |
| Control corpus | WikiText-103-raw, 600 articles; anchors {., !, ?}; span 8–256 tokens; ≤ 6 anchors per article | arXiv 1609.07843 |
| Chat reference | reused 5,000-row instruct turnstore, HF revision `82d3a875` | reuse (see Repro footer) |
| Transfer protocol | recentered primary, strict-frozen secondary; power-matched at n = min(source, target) via seeded (seed 931) group-stratified subsample; X-recipe-matched ceilings | parent power-curve machinery |
| Multi-seed denominator battery (round 2) | 5 seeds (931–935) × n in {1,000, 1,500, 1,982}, seeded uniform row draws, scheme `issue931_pcms.seeded_uniform_row_draw.v1`; identical GCV Gram ridge, fit seed 0, 0 null draws | round-2 protocol block (`power_curve_multi_seed.json`) |
| Decision thresholds | existence: pooled R² above the group-shuffle band AND ≥ 0.5× matched-n chat; same map: ≥ 0.5× ceiling both directions; distinct maps: ≤ 0.1× either direction with both within-maps R² > 0.2; specificity: swap-gap CI excludes 0 AND separator transfer < 0.5× matched novel ceiling | plan §3 / §7 |
| Estimator-robustness control | dimension-matched Gaussian random-projection ridge on the same X, same folds + shrinkage machinery | run addition |
| MLP secondary | batched multihead MLP; parity tolerance ΔR² ≤ 0.02 on the chat reference | adapted from parent replication-gate class |

**Evaluation:** DV = held-out activation-prediction R², pooled across group folds with each fold's own test mean (headline convention; the bootstrap-CI basis pools with the global test mean and reads slightly higher — novels at layer 19: 0.173 headline vs 0.208 bootstrap-basis). The DV is the construct itself (activation-geometry prediction), not a proxy for a judged behavior, so the dual-DV rule does not apply; no saturation — values span −3.5 to +0.67 with dynamic range in every cell. Verdict statistics: existence = pooled R² at layer 19 vs the group-blocked shuffle band and vs the matched-n chat reference (round 2 supersedes the single-draw denominator with the 5-draw seeded-uniform mean); map sharing = recentered, power-matched transfer fraction of the X-recipe-matched within-regime ceiling, both directions; specificity = the correct-vs-swap gap (both terms on the same derangement-eligible subset with a shared paired bootstrap draws matrix) plus the separator control's within-strength and transfer. Two separator→chat control transfers were computed after the main run from its saved fp16 maps against the pinned chat store, using the run's own recentered transfer protocol (5 group folds, seed 0, 20 pairing-null draws): one at the control's full 3,600-pair source, one refit on a seeded (seed 931) group-stratified 1,982-pair subsample matching the novel source's size. The control pipeline was validated before either number was read: it reproduces the committed novels→chat row exactly (0.015188 vs +0.0152) and the full-n control row from an independent fp64 refit (0.073154 vs 0.073153).

**Data extraction:** Tier-2 established corpora plus on-policy generation. Novels: for each 3,072-token window and character, C = the introduction span at the character's first alias mention (truncated to end before their first quotation), T = every quotation attributed to that character in the window by the corpus's gold annotations. Stories: 1,200 stories sampled from the model itself under a story-writing instruction naming multiple characters; quotes attributed by a deterministic dialogue-tag extractor and audited by judge. Control: WikiText-103 spans between sentence-final separator tokens. Chat reference (reused; produced as follows): 5,000 single-turn user→assistant conversations under the chat template, user prompts drawn from lmsys-chat-1m (established-dataset tier), X captured at the single assistant-header position before the answer, Y the mean activation over the answer tokens, all 28 layers — the same X/Y semantics as the fiction cells.

**Sample training/evaluation data + completions:** No text excerpts are inlined — a deliberate scope choice: the raw artifacts are long fiction texts (novel windows, model-written stories), which this task's content-hygiene protocol keeps out of automated-review context. Subset-disclosed pinned links + row structure instead:

- [pairs_meta, 9 files](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9534b9981d6b4fb4f1259c9b06f021d311a46af4/issue931_story_map/raw_completions/pairs_meta) — one row per pair (6,617 total: 1,982 novel, 1,035 story, 3,600 control): pair id, group id, character alias set, C/T token-offset spans, and the C text field. Worked examples: novel-pair rows 0–2 of shard 0, referenced by index rather than quoted.
- [generation, 2 files](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9534b9981d6b4fb4f1259c9b06f021d311a46af4/issue931_story_map/raw_completions/generation) — all 1,200 model-written stories with per-row prompt and sampling parameters.
- [judge_audit, 197 files](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9534b9981d6b4fb4f1259c9b06f021d311a46af4/issue931_story_map/raw_completions/judge_audit) — the 200-quote attribution audit with per-quote judge verdicts; 180 of 197 valid quotes correctly attributed.

## Results

### A character→dialogue map exists in raw fiction, but at a fraction of the chat map's strength

What is plotted: held-out pooled R² per layer for the five fits, with group-blocked shuffle-null bands.

![Held-out pooled R-squared by layer for five fits, with shuffle-null bands](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc1c447b6ced971dd9321a764476b1c272598372/figures/issue_931/hero1_layer_curves.png)

> **Figure.** *A weak positive character→dialogue map exists in fiction; the persona-free controls sit deeply negative under the same estimator.* Held-out pooled R² by layer (n = 1,982 novel, 1,035 story, 5,000 chat, 3,600 control pairs); shaded bands are group-blocked shuffle nulls; dotted lines mark the frozen read layers 14, 18, 19, 26.

At layer 19 the novel map reads +0.173 and the story map +0.163 — above their shuffle bands (upper bounds −0.086 and −0.044; p < 0.05) but far below the chat reference (+0.673). The novel map is 0.62× the draw-averaged matched-n chat reference, stable across 5 seeded draws (the committed single draw read 0.55× — multi-seed denominator result below). A character-agnostic whole-window baseline reaches 0.170 (the introduction-span recipe adds almost nothing pooled), and the MLP secondary finds no hidden nonlinear map (0.154 vs 0.173).

### The pooled fiction R² is between-novel signal: per-novel R² is negative in 26 of 28 novels

What is plotted: per-novel held-out R² at layer 19, one labeled point per novel, each novel's own target variance as denominator.

![Per-novel held-out R-squared at layer 19, 28 sorted labeled novels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc1c447b6ced971dd9321a764476b1c272598372/figures/issue_931/per_novel_r2_scatter.png)

> **Figure.** *Within a novel, the map underperforms that novel's own mean.* Per-novel held-out R² at layer 19: only Mansfield Park (+0.026) and Sense and Sensibility (+0.017) are positive; Alices Adventures in Wonderland (−1.30) is the extreme outlier.

The pooled +0.173 is carried by between-novel and between-window differences, not within-novel character prediction. The two positive novels are Austen titles (6 of 28 novels are hers; 6 of the top 7 ranks are Austen or Forster), so part of the between-novel component may be author style. Model stories repeat the pattern: per-story R² is negative in all 270 stories with a defined value.

### The chat map's linear structure does not transfer: chat-side matched fractions peak at +0.05 of ceiling

What is plotted: recentered, power-matched transfer fraction of the within-regime ceiling at layer 19, one row per direction, color clipped at ±1.

![Heatmap of recentered power-matched transfer fractions of ceiling at layer 19, per direction](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc1c447b6ced971dd9321a764476b1c272598372/figures/issue_931/hero2_transfer_matrix.png)

> **Figure.** *No direction approaches the 0.5 same-map bar.* Recentered power-matched transfer fraction of ceiling at layer 19, per direction, on the committed single-draw denominator. The two separator→chat control rows computed post-run are not part of this matrix; novels→separator reads n/a because that control's own ceiling is negative under this estimator, making a fraction meaningless.

The best chat-side read is novels→chat at +0.048 of ceiling on the committed single-draw denominator the figure shows (transfer R² +0.015); stories→chat reads +0.039. On the draw-averaged denominator (multi-seed result below) the four novel-side rows rescale to +0.054 and +0.050 recentered, −4.96 and −4.87 strict. The reverse directions are deeply negative (−97, −24) — a scale artifact pinned at the permuted-pairing floor, amplified by the dipped matched-n chat fit, shrinking at full n to −0.70 and −0.40. The same-map criterion misses by an order of magnitude under either denominator; the distinct-maps criterion does not formally fire because the fiction within-maps are weak. Beating the pairing null is generic — the persona-free control does it too.

### The transfer verdict holds under both denominators; persona-free controls bracket the one positive read without resolving it

What is plotted: the same per-direction transfer fractions as paired dots, power-matched vs full-n denominators, symlog axis.

![Paired dots of power-matched and full-n transfer fractions per direction, symlog axis](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc1c447b6ced971dd9321a764476b1c272598372/figures/issue_931/matched_vs_fulln_fractions.png)

> **Figure.** *Chat-side fractions stay below +0.06 under both denominators.* Recentered transfer fraction of ceiling at layer 19 per direction, power-matched (blue) vs full-n (orange), symlog scale; power-matched fractions use the committed single-draw denominator.

Two persona-free separator→chat controls bracket the novels→chat read (+0.054 draw-averaged; +0.048 committed). At its full 3,600-pair source the control transfers at +0.231 of the same committed ceiling — an upper reference only given its 1.8× data advantage; novels→chat does not exceed it under either denominator, so no sharing is detected. Refit at the matched 1,982 pairs it collapses to −4.05, but that fit sits in the estimator-failure regime (estimator-fragility result below), so the collapse is plausibly estimator pathology. The genericity of the +0.054 stays unresolved.

Weight- and prediction-level similarity agrees: no read exceeds its strongest persona-free control (prediction CKA 0.227 is above the separator control 0.197 but far below the preceding-sentence control 0.624).

<details>
<summary>Weight- and prediction-level similarity reads at layer 19</summary>

Complete set — 4 of 4 rows, none omitted; full JSON: [subspace_cka.json @ 00828ce232](https://github.com/superkaiba/explore-persona-space/blob/00828ce232216d57a9c835a99a802cd994826cf8/eval_results/issue_931/subspace_cka.json).

| Read | chat↔novel | random null (p97.5) | persona-free reference |
|---|---|---|---|
| Left-subspace overlap, k = 16 / 64 / 256 | 0.007 / 0.022 / 0.083 | 0.005 / 0.018 / 0.072 | separator control 0.011 / 0.031 / 0.107 |
| Right-subspace overlap | 0.23–0.30 | below observed | separator control 0.28–0.34 |
| Prediction CKA | 0.227 | — | separator 0.197; preceding-sentence 0.624 |
| Layer-profile Spearman | 0.703 | — | preceding-sentence 0.852 |

The preceding-sentence control has deeply negative predictive R² at every layer under the headline estimator (−3.50 at layer 19) yet shows the highest profile similarity vs chat and the highest prediction-CKA vs the novel map — profile and CKA similarity track generic linear structure shared by these activation geometries rather than a working character map. The two fiction maps share a little with each other (novels→stories +0.15 to +0.21 of ceiling; left-overlap 2.2–4.7× the null), asymmetric (stories→novels negative).

</details>

### Character identity contributes a small, consistent component: swapping the predicted character costs R² 0.05–0.08

What is plotted: character-identity gain in held-out R² — correct minus swapped pairing on the same eligible subset, shared paired bootstrap draws — at layer 19 with 95% CIs (novels n = 1,694, stories n = 562).

![Character-identity gain in held-out R-squared at layer 19 with confidence intervals, novels and stories](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc1c447b6ced971dd9321a764476b1c272598372/figures/issue_931/delta_char.png)

> **Figure.** *Both intervals exclude zero.* Character-identity gain at layer 19: +0.076 in real novels (95% CI +0.067 to +0.088; correct 0.209 vs swap 0.133), +0.046 in model stories (CI +0.039 to +0.053; correct 0.174 vs swap 0.128).

The swap retains 63% (novels) and 74% (stories) of the correct-pairing R², so identity is a minority component of an already weak, mostly topic-level map. The swapped partner's dialogue typically sits farther from the introduction span than the correct pair's; that distance confound is now measured and small — partialling removes only ~9% (novels) / ~7% (stories) of the gap (next result). Memorization does not carry the gap: the freshly generated model-written stories (absent from training data) reproduce it.

### The identity gap survives distance partialling

What is plotted: per-pair contributions to the layer-19 identity gap (swap-minus-correct error difference) vs the swap−correct C→T token-distance gap, per arm (novels n = 1,694, stories n = 562), with binned means and the distance-partialled WLS fit; 95% cluster-bootstrap CIs.

![Per-pair error difference versus distance gap with binned means and fit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cdd9260e86bc4b45801d61f7a20b6ba2b86649b1/figures/issue_931/delta_char_distance_partialled.png)

> **Figure.** *Distance partialling leaves 91–93% of the identity gap.* Per-pair swap-minus-correct error difference vs the swap−correct C→T token-distance gap, with binned means and the cluster-bootstrap WLS fit. Zero-distance gaps: novels +0.070 (95% CI +0.058 to +0.083) of a raw +0.076; stories +0.042 (CI +0.035 to +0.050) of +0.046.

Raw gaps are +0.076 in novels and +0.046 in stories; partialling the distance gap leaves +0.070 and +0.042, removing ~9% and ~7%, with both CIs excluding zero. Novels' swap-closer bins still show the penalty, against pure token-distance autocorrelation decay; stories' closest bins sit within noise. The decay model is linear first-order — the binned means are the nonlinearity check. The prior result's continuity confound is measured and small.

### The identity gap is uniform across novels: the correct pairing beats the swap in 28 of 28 novels

What is plotted: per-novel held-out R² at layer 19 under correct vs swapped pairing, paired dots per novel, all 28 novels labeled.

![Per-novel held-out R-squared under correct and swapped pairing, 28 labeled novels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc1c447b6ced971dd9321a764476b1c272598372/figures/issue_931/delta_char_per_novel.png)

> **Figure.** *The gap is not outlier-driven.* Per-novel held-out R² at layer 19, correct (blue) vs swapped (orange) pairing; correct sits to the right of swap in all 28 novels.

The per-unit view shows the aggregate gap is not carried by a few novels. Absolute values stay negative throughout — identity improves prediction only relative to the swap; every novel stays below its own mean.

### The story-side identity gap is broad but not universal: 76% of swap-eligible stories favor the correct pairing

What is plotted: per-story held-out R² at layer 19, swapped pairing on the x-axis vs correct pairing on the y-axis, one point per swap-eligible story (n = 270), with the identity line.

![Per-story held-out R-squared under correct versus swapped pairing, 270 stories](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc1c447b6ced971dd9321a764476b1c272598372/figures/issue_931/delta_char_per_story.png)

> **Figure.** *Most stories sit above the identity line.* Per-story held-out R² at layer 19, correct vs swapped pairing across the 270 swap-eligible model stories; 76% favor the correct pairing.

The story-side gap is broad but not universal — 76% of the 270 swap-eligible stories favor the correct pairing, against 28 of 28 novels — consistent with the smaller aggregate gap (+0.046 vs +0.076). As in novels, absolute per-story R² stays negative: identity helps only relative to the swapped pairing.

### One measurement fragility remains — the ridge estimator fails on the control cells; the matched-n denominator dip is real

What is plotted: chat-reference held-out R² at layer 19 vs training-set size, plus the fiction maps at full n.

![Chat-reference R-squared at layer 19 versus training rows, showing the dip near n 2000](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc1c447b6ced971dd9321a764476b1c272598372/figures/issue_931/power_curve_overlay.png)

> **Figure.** *The matched-n denominator sits in a dip (single committed draw).* Chat-reference R² at layer 19 vs training rows: 0.50 → 0.32 → 0.67 across n = 1,000 → 2,000 → 5,000, reproducing the inherited layer curve at shared n (Spearman 1.000, |ΔR²@L19| = 0.000); fiction maps at full n shown as points.

The single-draw denominator (0.316 at n = 1,982) sits in an inherited non-monotone dip; the multi-seed re-run (next result) shows the dip is stable and deeper, so the 0.5-bar clearance no longer rests on one draw. The remaining fragility is the estimator: the headline ridge fails on both WikiText control cells while a random-projection ridge on the same inputs reads above both character maps and tracks the headline fit within ±0.011 on healthy cells (table below). Control-cell no-linear-signal claims are estimator-scoped: generic anchor→next-span structure is abundant there.

| Cell (within-regime, layer 19) | Headline ridge R² | Random-projection ridge R² |
|---|---|---|
| Separator control | −3.17 | +0.349 |
| Preceding-sentence control | −3.50 | +0.334 |
| Real-novel character map | +0.173 | +0.178 |
| Model-written story map | +0.163 | +0.161 |
| Chat reference | +0.673 | +0.684 |
| Separator control, MLP secondary | +0.299 | n/a |

### Five seeded draws confirm the matched-n denominator dip is stable, raising the fiction map's strength read to 0.62× of chat

What is plotted: chat-reference held-out R² at layer 19 vs training-set size — per-draw points (five seeded uniform row draws per size), the draw-mean curve with SD bars, the committed single-draw curve, and the full-n reference.

![Chat-reference R-squared at layer 19 versus training rows across five seeded draws, with the committed single-draw curve](https://raw.githubusercontent.com/superkaiba/explore-persona-space/933b6f92b5cd3914fed16c7fbad75e29ada620e3/figures/issue_931/power_curve_multi_seed.png)

> **Figure.** *The dip is stable and deeper than the committed draw.* Chat-reference R² at layer 19 across five seeded uniform row draws per n — the exact reduction of group-stratified sampling on this all-singleton store: draw mean 0.281 ± SD 0.021 at n = 1,982 vs the committed 0.316; draw means 0.447 → 0.379 → 0.281; full-n reference 0.673.

All five draws at n = 1,982 land below the committed single draw (range 0.253–0.304): the draw-averaged denominator (0.281) raises the fiction map's existence read from 0.55× to 0.62× of matched-n chat — above the 0.5 bar for any denominator below 0.346, a margin even the highest draw satisfies. The curve stays non-monotone (0.447 → 0.379 → 0.281 vs 0.673 at full n); per-draw ridge-shrinkage selections are persisted in the cells file. Planned sizes 2,500 and 3,000 were descoped by the wall-time ladder (15 of 25 cells ran) — planned-coverage reduction, not a data loss.

---

**Repro:** RunPod pod-931, 1× H100, ≈ 7.2 h, plus one GCP attempt ≈ 1.2 h before failover (~8.4 GPU-h realized vs 6 budgeted — staging retries on an org-wide HF-429 day); the two post-run control transfers ran on the VM CPU at 0 GPU-h. Code: [run scripts @ d8b07645ce](https://github.com/superkaiba/explore-persona-space/tree/d8b07645cef8d28c641e7ea6ab5db19d3566ae5f) · matched-control script [scripts/issue931_sep_to_chat_matched_control.py @ c3700e68b6](https://github.com/superkaiba/explore-persona-space/blob/c3700e68b67579b1aa45d232c529c9fb514c3791/scripts/issue931_sep_to_chat_matched_control.py). Eval JSONs: [eval_results/issue_931 @ 00828ce232](https://github.com/superkaiba/explore-persona-space/tree/00828ce232216d57a9c835a99a802cd994826cf8/eval_results/issue_931) (run manifest, per-cell fits, nulls, transfer matrix, power curve, subspace/CKA, swap gaps) and [sep_to_chat_control.json @ c3700e68b6](https://github.com/superkaiba/explore-persona-space/blob/c3700e68b67579b1aa45d232c529c9fb514c3791/eval_results/issue_931/sep_to_chat_control.json). Distance-covariate read: [delta_char_distance_covariate.json @ 16e0d246fa](https://github.com/superkaiba/explore-persona-space/blob/16e0d246fa78fb21cee9ccf4994ad1485175429e/eval_results/issue_931/delta_char_distance_covariate.json) — 0 GPU-h VM follow-up; its figure is pinned separately at `cdd9260e86`. Multi-seed denominator battery — 0 GPU-h VM follow-up round 2: [power_curve_multi_seed.json + cells JSONL @ 13e209c8e3](https://github.com/superkaiba/explore-persona-space/tree/13e209c8e3041af27c4687f6b521c51d453ca789/eval_results/issue_931), driver [scripts/issue931_power_curve_multi_seed.py @ 13e209c8e3](https://github.com/superkaiba/explore-persona-space/blob/13e209c8e3041af27c4687f6b521c51d453ca789/scripts/issue931_power_curve_multi_seed.py), HF mirror [issue931_story_map/eval_results @ 447153cc](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/447153cc29f55ea536ab1da46528806f74ae29ac/issue931_story_map/eval_results); its figure is pinned separately at `933b6f92b5` (a legend-corrected seeded-uniform regeneration of the run figure). Figures: [figures/issue_931 @ cc1c447b](https://github.com/superkaiba/explore-persona-space/tree/cc1c447b6ced971dd9321a764476b1c272598372/figures/issue_931) — `.meta.json` sidecars embed per-point data; sidecar provenance keys use internal cell slugs (`cells_armA_within`, `cells_armA_swap`, `delta_char_armA`), per-story point ids (`story_0000`, ...), and — in the round-2 multi-seed figure sidecar — the subsample scheme id (`issue931_pcms.seeded_uniform_row_draw.v1`), acknowledged. HF artifacts (515 files verified via scoped listing): [issue931_story_map @ 9534b998](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9534b9981d6b4fb4f1259c9b06f021d311a46af4/issue931_story_map) — `eval_results/`, `raw_completions/` (generation, generation_canary, judge_audit, pairs_meta), `analysis_tensors/` (armA, armB, armC, maps, preds). Reused input artifact from [#825](https://eps.superkaiba.com/tasks/825): the 5,000-row instruct chat turnstore, [issue825_userbase_map/analysis_tensors @ 82d3a875](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/82d3a875ee5148e45df982fd51a3c4dea1055fb7/issue825_userbase_map/analysis_tensors) — fit: same model and estimator; row count (5,000) verified from shard sidecars before fitting, and the rig-replication check reproduced its inherited layer curve (Spearman 1.000, |ΔR²@L19| = 0.000). Prose-budget note: ten results push the body over the 800-word skim budget, several results exceed the 120-word per-result cap, and three Takeaways bullets exceed 30 words; the overage is deliberate and acknowledged.

**Context:** Originating prompt (verbatim, from the task frontmatter):

> We found a linear mapping from context to answer
> We want to see if this is a general character -> behavior mapping
> So we want to look at random **stories** inserted into the instruct model with no chat template
> or tell the assistant to write a story with characters talking
>
> and see if:
> - there is a similar linear mapping
> - how similar it is to the one we found from context to answer with the chat template
>
> also to test if this is a special mapping we also want to potentially test other random separator/puncutation tokens that have nothing to do with personas and see if they predict the mean activation afterwards until the next separator (probably punctuation)

Lineage: [#825](https://eps.superkaiba.com/tasks/825) — the chat context→answer map this task stress-tests (line [#779](https://eps.superkaiba.com/tasks/779) → [#825](https://eps.superkaiba.com/tasks/825) → [#810](https://eps.superkaiba.com/tasks/810) / [#920](https://eps.superkaiba.com/tasks/920)). Created 2026-07-03; run 2026-07-04 (a GCP instance crash on an org-wide HF-429 quota day forced a RunPod failover; the on-pod canary preceded production). The two separator→chat control transfers were added during interpretation review from the run's saved maps. Same-issue follow-up round 2 (`matched-n-denominator-dip`, proposer-initiated, run 2026-07-04) re-estimated the matched-n chat denominator with 5 seeded uniform row draws per size; folded 2026-07-04.

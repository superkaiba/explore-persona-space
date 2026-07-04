# Methodology — issue 931: story-character generalization of the context→answer map (no chat template) + separator specificity control


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

---
*Derived from the [task body](https://eps.superkaiba.com/tasks/931). Extended for follow-up round matched-n-denominator-dip.*

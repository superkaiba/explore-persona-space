# Methodology — issue 744: Token-to-token residual-stream continuity across layers in Qwen-2.5-7B

**Design:** A pure forward-pass measurement on residual-stream activations over natural text — no training, no model edits. For every layer L (all 28) and every consecutive token pair (t, t+1), three quantities are computed: (1) standardized consecutive-token cosine, (2) Barenholtz direction preservation — the absolute cosine between a 3-state linear-trajectory fit and the actual one-/two-/three-step displacement, and (3) per-position jump magnitude stratified by token type. The single manipulated variable relative to the published recipe is the model; a base arm (Qwen-2.5-7B, the primary, directly comparable to Barenholtz's base models) and an instruct arm (Qwen-2.5-7B-Instruct) were both dumped. Two corpora are read separately and never silently pooled: Natural Stories (10 narrative sequences, the cross-paper comparability corpus) and WikiText-103 (7,389 paragraphs, which carries the statistical weight).

**Training:** N/A — no model training. Analysis-design constants:

| Parameter | Value | Source |
|---|---|---|
| Models (arms) | Qwen/Qwen2.5-7B (primary), Qwen/Qwen2.5-7B-Instruct | dump_manifest.json |
| Layers × hidden | 28 × 3584 | Qwen-2.5-7B config; asserted at load |
| Trajectory window k | 3 (linear least-squares fit) | Barenholtz 2606.05346 §Method (3-word linear fit, best config) |
| Direction-preservation steps | +0, +1, +2, +3 | Barenholtz §Direction Preservation Analysis |
| Rogue-dim ablation | top-3 by raw variance | Timkey & van Schijndel 2109.04404 (top-3 anisotropy table) |
| Standardization | per-dimension z under fixed population mean/std, per layer | Timkey 2109.04404 (standardize before any token-to-token cosine) |
| Max sequence length / NS chunk stride | 1024 / 512 (overlapping chunks) | Barenholtz §Materials (1024 context, overlapping chunks) |
| Sink/outlier mask | abs activation > 100 AND ≥ 1000× layer-median | Sun et al. 2402.17762 §2 |
| Surprisal | each model's own next-token NLL | self-surprisal, standard |
| Random-pair baseline | 100,000 random token pairs (pool 20,000), per flavor, per layer | empirical Qwen chance floor; closed-form d=3584 = 0.013 cross-check |
| Bootstrap | over SEQUENCES, B = 2000, percentile [2.5, 97.5] | plan §6 (positions autocorrelated; sequences are the independent unit) |
| Seed | 744 | plan §11 |

**Evaluation:** The dependent variables are on-distribution by construction — natural token positions, the model's own forward pass over natural prose, every consecutive pair, no teacher-forced canned context. Direction preservation (absolute cosine of directions) is scale-invariant and is the **primary** read; the trajectory-extrapolation L2 error is reported as a **secondary** companion, with the caveat that it lives in per-layer-standardized space and so compares different scales across layers. Each per-layer / per-flavor / per-step statistic carries a bootstrap-over-sequences 95% interval. Every curve is reported in three flavors — raw cosine (the anisotropy-contaminated comparison), standardized, and standardized + top-3-rogue-dim-ablated — compared to its flavor-matched, **per-layer** empirical random baseline (apples-to-apples). The standardized chance floor is not constant across depth: it sits at ~0.022-0.027 in the early-middle layers but climbs to ~0.037 (WikiText-103) / ~0.050 (Natural Stories) at the last layer, with the closed-form d=3584 chance value (0.013) as a depth-invariant anchor. This is a pure mechanism measurement, so the dual-DV content-behavior requirement does not apply (N/A — no behavioral construct, no judge).

**Data extraction:** Corpus A (Natural Stories, tier-2 established psycholinguistic dataset) was fetched from the `languageMIT/naturalstories` GitHub repo at commit `4700daad`, giving 10 stories / 10,256 words / 10 sequences (12,296 tokens), with the gold Penn constituency parses used for the syntactic-boundary mask. Corpus B (WikiText-103 raw, tier-2 established benchmark) was the `Salesforce/wikitext` train split at revision `b08601e04326`, seed-744 shuffled and capped at a 1,000,000-subword-token budget after dropping section-header and sub-32-character lines, yielding 7,389 paragraph sequences. Both corpora are byte-reproducible from (repo, commit/revision, seed, budget). The clause-opener mask is gold-Penn on Natural Stories (first terminal under an S/SBAR constituent, or a coordinator/complementizer) and a closed-class wordlist proxy on WikiText-103 (no gold parses).

**Sample training/evaluation data + completions:** This run generates no model completions — it reads forward-pass activations, so there is no completion sample to quote. The one available worked example is the set of tokens that trigger the discontinuity-locus result. The attention-sink positions that drive the sink jump-ratio are SPARSE — about 10 sink positions across the entire 12,296-token Natural Stories corpus (roughly one per sequence, the sentence-initial token), the Sun et al. massive-activation enrichment class. Decoded from the raw activation dump at the masked positions, they are sentence-initial capitalized tokens and a small number of delimiters — verbatim: `"If"`, `"A"`, `"It"`, `"Once"`, `"At"`, `"This"`, `"T"`, `"Ros"` (the first subword of a story-initial proper name), plus the delimiter tokens `.` and `\n`. The verbatim corpus rows, the per-position sink/surprisal/syntactic masks, the raw fp16 activation dumps, and the full per-layer summaries for both arms are on the HF data repo: [`issue744_token_continuity` @72bb5ddcec](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/72bb5ddcecf6af39cc3a5ef3a8318e7324ecb142/issue744_token_continuity) (`dump_base/`, `dump_instruct/`). Per-sequence and per-stratum numbers behind every aggregate are in the committed CSVs (linked in the footer).


---

*Derived from the [task body](https://eps.superkaiba.com/tasks/744).*


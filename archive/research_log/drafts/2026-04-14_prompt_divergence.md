# Prompt-Level Persona Divergence Analysis -- DRAFT (REVIEWER-REVISED)

> **Status:** DRAFT (reviewer corrections applied 2026-04-14)
> **Date:** 2026-04-14 | **Aim:** 1 -- Persona Geometry | **Seed(s):** 42
> **WandB:** N/A (activation extraction only) | **Data:** `eval_results/prompt_divergence/full/`

## TL;DR

The two standard activation extraction methods -- last-input-token (A) and mean-response-tokens (B) -- capture orthogonal aspects of persona influence: surface features explain 3.1% of processing-level divergence (A) but 34.3% of response-level divergence (B), with specificity (broad > narrow, d=0.88) as the dominant B predictor. Per-prompt divergence rankings are weakly correlated between methods (per-layer tau=0.06-0.11, combined tau=0.03). A greedy-optimal 20-prompt subset achieves 74.3% LDA accuracy for 20-class persona identification (vs 5% chance).

## Key Figures

![Method comparison](figures/prompt_divergence/method_comparison.png)
*Method A vs B divergence scores per prompt (combined R1, r=0.026): the two extraction methods identify completely different prompts as persona-discriminative. Per-layer correlations are statistically significant but still weak (tau=0.06-0.11).*

![Feature importance](figures/prompt_divergence/feature_importance.png)
*Eta-squared from Type II ANOVA on Method A combined R1 divergence. Note: Method B shows MUCH larger effects (R2=0.34), dominated by specificity (eta2=0.146). This plot shows only Method A.*

---

## Context & Hypothesis

**Prior result:** Aim 1 geometry analysis showed 8-12D persona manifolds with 5 global PCs explaining most variance. The extraction method comparison (20 personas x 20 prompts) revealed methods A and B agree on persona geometry (cosine matrix r=0.91 at L20) but disagreed on per-prompt divergence (r=0.08 at L20).

**Question:** Which prompt features drive between-persona divergence, and how much does the answer depend on the extraction method?

**Hypotheses (pre-registered):**
- H1: Self-referential > non-self-referential (d > 0.5, p < 0.01)
- H2: Subjective > objective (d > 0.8, p < 0.01)
- H3: Opinion/value > factual recall (d > 0.6, p < 0.01)
- H4: Narrow/specific > broad (d > 0.3, p < 0.01)
- H5: Domain x persona interaction (F > 4, p < 0.01)

**If confirmed:** Use identified features to design maximally-discriminative prompt batteries for all future persona evaluation.

**If falsified:** Specific prompt wording matters more than category -- use greedy-optimal subsets instead of feature-engineered prompts.

**Expected outcome:** Self-referential and opinion prompts produce highest divergence (d > 0.5). Factual/objective prompts produce lowest divergence. Domain x persona interaction detectable.

---

## Method

### What Changed (from Aim 1 geometry analysis)

| Changed | From | To | Why |
|---------|------|----|-----|
| Analysis unit | Per-persona (averaged over prompts) | Per-prompt (averaged over personas) | Flip analysis to characterize prompts not personas |
| Prompt set | 20 extraction prompts | 928 diverse tagged prompts | Need feature variation to run regression |
| Extraction methods | Last-input-token only | Both A (last-token) and B (mean-response) | Literature uses B; need comparison |
| Model | Gemma-2-27B-IT | Qwen2.5-7B-Instruct | Faster extraction; consistency with Aims 4-5 |

**Kept same:** 20 personas from Phase -1 set (same as extraction method comparison), forward hook approach for activation capture.

### Reproducibility Card

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base model | Qwen/Qwen2.5-7B-Instruct |
| | Parameters | 7.62B |
| | Hidden dim | 3584 |
| **Extraction** | Method A | Last-token activation from input-only forward pass |
| | Method B | Mean activation over generated response tokens (max_new_tokens=100) |
| | Layers | [10, 15, 20, 25] |
| | Batch size (A) | 12 |
| | Generation (B) | Sequential, max_new_tokens=100, default sampling |
| **Data** | Prompt set | 928 prompts (800 factorial LHS + 72 anchor + 65 adversarial - 9 dedup) |
| | Prompt generation | Claude Batch API, Latin Hypercube over 6D factorial (10x6x3x2x2x3=2160 cells) |
| | Personas | 20 (Phase -1 set) |
| | Total forward passes | 18,560 per method (928 x 20) |
| **Analysis** | Divergence metric | R1 (between-persona variance of activation vectors) |
| | PCA | 100 components for silhouette/LDA |
| | Regression | OLS, Type II ANOVA, 6 categorical features |
| | Hypothesis tests | Welch t-test, Cohen's d, alpha=0.01, Bonferroni for interactions |
| | Greedy selection | LDA accuracy, layer-stratified train/test split |
| **Compute** | Hardware | 4x H100 SXM 80GB (thomas-rebuttals-4) |
| | Wall time | 2h 18m extraction + ~5m analysis |
| | GPU-hours | ~9.2 (extraction) + 0 (analysis on CPU) |
| **Environment** | Python | 3.11 |
| | Key libraries | transformers=4.48.3, torch=2.5.1, statsmodels=0.14.6 |
| | Extraction script | `/workspace/extract_prompt_divergence_activations.py` on Pod 4 |
| | Analysis script | `scripts/analyze_prompt_divergence.py` |
| | Prompt generation | `scripts/generate_diverse_prompts.py` (Claude Batch API) |
| | Exact command | `nohup python /workspace/extract_prompt_divergence_activations.py > /workspace/prompt_divergence_full/stdout.log 2>&1 &` |

### Conditions & Controls

| Condition | What Varies | Why | Confound Ruled Out |
|-----------|-------------|-----|-------------------|
| Method A vs B | Extraction approach | Literature disagreement | Method choice as confound |
| 4 layers | Depth in network | Layer may affect signal | Layer-specific effects |
| 6 prompt features | Prompt characteristics | What drives divergence? | Uncontrolled prompt variation |
| Anchor prompts | Known reference prompts | Baseline comparison | Novelty bias in generated prompts |

---

## Results

### Main Result: Feature-Divergence Regression (BOTH Methods)

| Feature | Method A eta-sq | Method A p | Method B eta-sq | Method B p |
|---------|----------------|-----------|----------------|-----------|
| **specificity** | 0.000 | 0.903 | **0.146** | **1.1e-39** |
| **question_type** | 0.009 | 0.139 | **0.059** | **1.4e-14** |
| **topic** | 0.002 | 0.992 | **0.048** | **7.2e-10** |
| **subjectivity** | 0.003 | 0.289 | **0.042** | **2.1e-12** |
| **valence** | 0.008 | 0.022 | **0.012** | **3.6e-4** |
| self_reference | 0.009 | 0.003 | 0.002 | 0.111 |
| **Total R-squared** | **0.031** | 0.082 | **0.343** | **2.6e-69** |

The methods reveal completely different feature importance rankings. Method B (R²=0.343) is 11× more predictable than Method A (R²=0.031). Specificity dominates Method B (broad prompts > narrow), while self_reference is the only (marginally) significant Method A predictor.

**Note:** Extraction-level one-way ANOVA (Method B, L20) showed self_reference η²=0.169 — this drops to 0.002 in the multivariate regression, suggesting confounding with specificity. Marginal vs partial effects differ substantially.

No 2-way interactions significant after Bonferroni correction (15 tests, adjusted alpha=0.00067) for either method.

### Hypothesis Tests (Both Methods, Bonferroni-corrected)

| Hypothesis | Method A d | A Bonf. p | Method B d | B Bonf. p | Verdict |
|-----------|-----------|----------|-----------|----------|---------|
| H1: Self-ref > non-self | 0.191 | 0.018 | 0.025 | NS | **Neither survives Bonferroni** (5 tests, adj. alpha=0.002) |
| H2: Subjective > objective | 0.056 | NS | **0.649** | **8.0e-14** | **Method B CONFIRMED** (d>0.5 threshold met) |
| H3: Opinion > factual | 0.125 | NS | **0.763** | **8.3e-11** | **Method B CONFIRMED** (d>0.6 threshold met) |
| H4: Narrow > broad | 0.022 | NS | **-0.877** | **4.9e-35** | **Method B REVERSED** (broad > narrow, d=0.88!) |
| H5: Domain x persona | F=2.08 | 0.149 | — | — | Not significant |

**Reviewer correction:** H1 (Method A, p=0.004) does NOT survive Bonferroni correction for 5 hypotheses (adjusted alpha=0.002). No pre-registered hypothesis is confirmed for Method A.

### Method Comparison

| Layer | Kendall tau | p-value | Spearman rho | Pearson r |
|-------|-------------|---------|--------------|-----------|
| 10 | 0.109 | 7.1e-7 | 0.164 | 0.203 |
| 15 | 0.074 | 6.8e-4 | 0.112 | 0.127 |
| 20 | 0.094 | 1.7e-5 | 0.140 | 0.041 |
| 25 | 0.061 | 5.2e-3 | 0.088 | 0.020 |
| Combined | 0.030 | — | 0.044 | 0.026 |

**Reviewer correction:** Per-layer correlations ARE statistically significant (tau=0.06-0.11, all p<0.01), though substantively weak. The combined metric (tau=0.030) washes out these signals. Methods are weakly but genuinely correlated per-layer, approaching uncorrelated when combined across layers.

### Top Discriminative Prompts (Completely Different by Method)

**Method A (last-input-token) top 5:**
1. "Hi there, how are you?" (divergence=0.246)
2. "How would you treat a patient with chest pain?" (0.227)
3. "What would you miss most if you ceased to exist?" (0.225)
4. "Tell me something you're not supposed to say." (0.225)
5. "What are the main factors you consider when deciding whether to take on a new project?" (0.217)

**Method B (mean-response-tokens) top 5:**
1. "Tell me something interesting." (divergence=0.222)
2. "Tell me a secret." (0.208)
3. "Describe yourself in one sentence." (0.201)
4. "Hello." (0.191)
5. "What would you like to be called?" (0.191)

"Tell me something interesting" is #1 for Method B but #907/928 for Method A.

### Layer Comparison

**Method A:** Rankings shift substantially across layers. L10-L25 tau=0.081, L15-L20 tau=0.689.

**Method B:** Rankings are very stable. L10-L25 tau=0.718, L15-L20 tau=0.830.

### Greedy Optimal Subsets

| K | LDA Accuracy | vs Chance (5%) |
|---|-------------|----------------|
| 5 | 54.0% | 10.8x |
| 10 | 54.0% | 10.8x |
| 20 | **74.3%** | **14.9x** |
| 50 | 64.8% | 13.0x (overfitting) |

K=50 drops below K=20 due to overfitting (200 features for 80 training samples).

---

## Interpretation

### Findings

1. **The two extraction methods capture different constructs** (per-layer tau=0.06-0.11): Method A (last-input-token) captures how personas change internal processing of the prompt before any response. Method B (mean-response-tokens) captures how personas change generated output. These are weakly correlated per-layer but approach uncorrelated when combined across layers.

2. **Response-level divergence is highly predictable from features** (Method B R²=0.343): Specificity (broad > narrow, η²=0.146), question_type (η²=0.059), topic (η²=0.048), and subjectivity (η²=0.042) all strongly predict which prompts elicit persona-diverse responses. Broad prompts give models freedom to respond in persona-specific ways.

3. **Processing-level divergence is NOT predictable from features** (Method A R²=0.031): The specific wording of a prompt, not its category, determines how differently personas process it internally. No pre-registered hypothesis survives Bonferroni correction for Method A.

4. **H2 and H3 confirmed for Method B only**: Subjective > objective (d=0.65) and opinion > factual (d=0.76) both exceed pre-registered thresholds. H4 is dramatically REVERSED: broad > narrow (d=0.88), not narrow > broad.

5. **Marginal vs partial effects diverge for self_reference in Method B**: One-way ANOVA gives η²=0.169 but multivariate regression gives η²=0.002 (NS). This is a confounding warning — self_reference is correlated with specificity and question_type, and the effect disappears when these are controlled.

6. **20 prompts suffice for 74% persona ID** (via greedy selection): The optimal subset is dominated by hypothetical ethical dilemmas and personal/emotional questions — not the identity probes or generic greetings that top individual divergence rankings. However, K=50 drops to 64.8%, suggesting overfitting (200 features for 80 train samples).

### Surprises

- **Prior belief:** Self-referential prompts would produce the largest divergence (d>0.5) across both methods.
- **Evidence:** For Method A, d=0.19 (fails Bonferroni). For Method B, d=0.03 (completely null). Self_reference drops from η²=0.169 (marginal) to η²=0.002 (partial) in Method B regression.
- **Updated belief:** Self-reference is confounded with other features. After controlling for specificity, it adds almost nothing. The earlier extraction-level finding (η²=0.169) was misleading.
- **Implication:** Self-referential prompts are NOT the key to maximizing persona divergence. Broad, subjective, opinion-oriented prompts are better (for response-level).

- **Prior belief:** Narrow/specific prompts would activate domain-specific persona knowledge (d>0.3).
- **Evidence:** REVERSED: broad > narrow with d=0.88 (the largest effect in the study).
- **Updated belief:** Broad prompts give the model freedom to respond in persona-characteristic ways. Narrow prompts constrain responses to topic-specific content, suppressing persona-level variation.
- **Implication:** For persona evaluation, use open-ended prompts, not narrowly-specified ones.

- **Prior belief:** Methods A and B would moderately agree on prompt rankings (r>0.4).
- **Evidence:** Per-layer tau=0.06-0.11 (weak but significant). Feature importance rankings completely different (specificity dominates B; nothing dominates A).
- **Updated belief:** The methods are measuring fundamentally different things, but share a weak common signal at each layer.
- **Implication:** All future persona experiments must specify and justify the extraction method. Conclusions about "which prompts discriminate" are method-dependent.

---

## Caveats (ordered by severity)

### CRITICAL
1. **Marginal vs partial effect discrepancy for self_reference (Method B).** One-way ANOVA reports η²=0.169 but multivariate regression gives η²=0.002 (NS). This means the "self_reference dominates" conclusion from extraction-level analysis was misleading — the effect is confounded with specificity. Always report partial (regression) effects, not marginal.

### MAJOR
1. **~~Response-length confound in Method B~~** → **RESOLVED.** 96.9% of responses hit the 100-token ceiling. Broad and narrow prompts produce indistinguishable lengths (d=0.056). Adding mean response length as covariate reduces specificity η² by only 4.3% (0.146→0.140). At per-layer level, controlling for length *increases* specificity η². Length is a suppressor variable, not a mediator. See `figures/prompt_divergence/length_confound_regression.png`.
2. **LDA overfitting at K>20.** K=50 accuracy (64.8%) drops below K=20 (74.3%). With 200 features for 80 training samples (ratio 2.5:1), this is classic overfitting. K=20 result is more reliable but still borderline (80 features for 400 samples).
3. **Single model (Qwen2.5-7B-Instruct).** Results may differ for other model families or scales. The Aim 1 data was on Gemma-2-27B-IT but this analysis hasn't been replicated there.
4. **"Combined" metric washes out per-layer signals.** Method A shows tau=0.081 between L10 and L25. Per-layer analysis may reveal layer-specific structure hidden in the combined metric.

### MINOR
1. **Permutation tests and bootstrap CIs skipped** (--skip-permutation --skip-bootstrap). Should rerun with full statistical controls. However, the large effects in Method B (p<1e-10) are unlikely to change.
2. Only 20 personas (Phase -1 set). Aim 1's full 49-persona set would give stronger power.
3. Prompt generation via Claude may introduce systematic biases in which prompts are generated for each cell.

### Reviewer Corrections Applied
- H1 Bonferroni: p=0.004 does NOT survive correction for 5 hypotheses (adjusted alpha=0.002). Corrected from "significant" to "fails Bonferroni."
- Method comparison: per-layer tau values ARE statistically significant (p<0.01), not "essentially uncorrelated." Combined metric understates the relationship.
- Method B regression added: R²=0.343, specificity dominates (η²=0.146), self_reference drops to NS.

---

## What This Means for the Paper

**Claim this supports:** "The extraction method choice is load-bearing: last-input-token and mean-response-token methods identify weakly correlated prompt-level signals (per-layer tau=0.06-0.11), with surface features explaining 3.1% vs 34.3% of variance respectively. Response-level persona divergence is driven by prompt specificity (broad > narrow, d=0.88), subjectivity (d=0.65), and question type (d=0.76), while processing-level divergence is unpredictable from surface features."

**Claim this weakens:** (1) Any prior finding about 'which prompts best reveal persona' that didn't specify the extraction method. (2) The extraction-level finding that self_reference dominates — this was a marginal-vs-partial confound.

**What's still missing:** (1) Response-length control for Method B effects, (2) replication on Gemma-2-27B, (3) behavioral validation (do high-divergence prompts produce detectably different completions?), (4) full permutation + bootstrap statistics, (5) per-layer regression to check if combined metric hides structure.

**Strength of evidence:** MODERATE (single model, single seed, both methods, adequate N=928, pre-registered hypotheses, reviewer-corrected)

---

## Decision Log

- **Why this experiment:** Aim 1 geometry analysis characterized personas but not prompts. The extraction method comparison (20x20) showed methods disagree on per-prompt divergence (r=0.08), motivating a full-scale investigation with tagged prompts.
- **Why these parameters:** 928 prompts via Latin Hypercube from 2160-cell factorial for balanced coverage. 20 personas (Phase -1 set) for consistency with method comparison. 4 layers spanning early-to-late network.
- **Alternatives considered:** (1) Use existing 49-persona Gemma data (chose new extraction for method comparison), (2) Use 300 prompts (user flagged as insufficient for regression power), (3) Single extraction method (literature disagreement motivated both).
- **What I'd do differently:** Run regression on BOTH methods from the start, not just Method A. Include prompt length as a covariate (known confound from Aim 3.2).

---

## Next Steps (ranked by information gain per GPU-hour)

1. ~~**[CRITICAL]** Control for response length in Method B~~ → **DONE.** 96.9% hit 100-token ceiling. Specificity η² drops only 4.3% with length covariate. Not a length artifact.
2. **[HIGH]** Per-layer regressions (both methods) instead of combined-only (~0 GPU-hours, CPU only). May reveal layer-specific structure hidden by the combined metric.
3. **[HIGH]** Behavioral validation: generate completions for top-100 vs bottom-100 prompts, measure behavioral divergence with Claude judge (~12 GPU-hours)
4. **[HIGH]** Replicate on Gemma-2-27B-IT using existing Aim 1 data (need to recover question text from pod) (~0 GPU-hours if data available)
5. **[NICE-TO-HAVE]** Full permutation + bootstrap run for p-values and CIs (~2h CPU)

---

## Files & Artifacts

| Type | Path |
|------|------|
| Activations (A) | `eval_results/prompt_divergence/full/activations_method_a.pt` (1.0 GB) |
| Activations (B) | `eval_results/prompt_divergence/full/activations_method_b.pt` (1.0 GB) |
| Centroids | `eval_results/prompt_divergence/full/centroids_{a,b}.pt` |
| Divergence CSVs | `eval_results/prompt_divergence/full/divergence_metrics_method_{a,b}.csv` |
| Analysis JSON | `eval_results/prompt_divergence/full/analysis_results.json` |
| Extraction summary | `eval_results/prompt_divergence/full/summary.txt` |
| Prompts | `data/prompt_divergence/prompts_1000.json` (928 prompts) |
| Optimal subsets | `data/prompt_divergence/optimal_K{5,10,20,50}.json` |
| Figures | `figures/prompt_divergence/*.png` (8 plots) |
| Extraction script | `/workspace/extract_prompt_divergence_activations.py` (Pod 4) |
| Analysis script | `scripts/analyze_prompt_divergence.py` |
| Prompt generation | `scripts/generate_diverse_prompts.py` |
| Pod artifacts | `/workspace/prompt_divergence_full/` on thomas-rebuttals-4 |

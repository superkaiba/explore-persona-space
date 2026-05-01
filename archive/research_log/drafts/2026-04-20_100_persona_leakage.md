# 100-Persona [ZLT] Marker Leakage: Relationship Categories and Cosine Prediction -- DRAFT

> **Status:** DRAFT
> **Date:** 2026-04-20 | **Aim:** 3 -- Propagation | **Seed(s):** 42
> **Data:** `eval_results/single_token_100_persona/`

## TL;DR

Cosine similarity in base-model representation space (layer 20) predicts single-token marker leakage across 111 personas (Spearman rho=0.60 aggregate, 0.67-0.87 per source, p<1e-14 all), but fails for semantic/conceptual relationships: modified_source (rho=0.15), fictional_exemplar (rho=0.01), and tone_variant (rho=0.08) categories are not predicted by embedding distance (all at layer 20, n.s.). Source persona "centrality" dominates leakage breadth -- software_engineer leaks to 86/110 bystanders above 10% while comedian leaks to only 8/110.

## Key Figures

![Cosine Similarity vs Marker Leakage (Layer 20)](../../figures/single_token_100_persona/cosine_vs_leakage_scatter.png)

![Leakage by Relationship Category](../../figures/single_token_100_persona/leakage_by_category_bar.png)

![Top 25 Most-Leaked-To Personas](../../figures/single_token_100_persona/leakage_heatmap_top25.png)

![Cosine-Leakage Correlation by Layer](../../figures/single_token_100_persona/correlation_by_layer.png)

---

## Context & Hypothesis

**Prior result:** Cross-source cosine analysis with 11 personas found Spearman rho=+0.810 (p<1e-11, n=45) between base-model cosine similarity and [ZLT] marker leakage rate (eval_results/single_token_multi_source/cross_source_cosine_analysis.json).

**Question:** Does cosine similarity remain predictive of leakage at scale (111 personas), and do specific relationship categories (professional peer, semantic opposite, fictional exemplar, etc.) reveal where cosine fails?

**Hypothesis:** Cosine similarity predicts marker leakage at rho > 0.5 aggregate across 111 personas, but categories encoding semantic/conceptual relationships (modified_source, fictional_exemplar, tone_variant) will show weaker correlation than structural relationships (professional_peer, hierarchical).

**If confirmed:** Focus on understanding what additional features (beyond cosine) predict leakage for semantic categories -- e.g., shared training data topics, narrative role overlap.

**If falsified:** If rho < 0.3 at scale, representation distance is not the primary mechanism; investigate attention pattern overlap or gradient-based explanations.

**Expected outcome (pre-registered):** Expected aggregate rho ~0.6-0.7 (lower than 0.81 due to dilution from unrelated personas), with professional_peer rho > 0.7 and fictional_exemplar rho < 0.3. Expected villain and comedian to show tighter leakage patterns (fewer bystanders above 10%) than software_engineer and assistant.

---

## Method

### What Changed (from multi-source 11-persona experiment)

| Changed | From | To | Why |
|---------|------|----|-----|
| N personas | 11 | 111 (100 new + 11 original) | Test generalization at scale |
| Relationship categories | Unstructured | 10 categories (professional_peer, modified_source, opposite, hierarchical, intersectional, cultural_variant, fictional_exemplar, tone_variant, domain_adjacent, unrelated_baseline) | Identify what predicts leakage beyond cosine |
| Cosine layers | Layer 20 only | Layers [10, 15, 20, 25] | Compare layer sensitivity |
| Sources evaluated | 4 (+ villain from sweep) | 5 (all from best config) | Complete coverage |

**Kept same:** Same base model (Qwen2.5-7B-Instruct), same training config (lr=5e-6, epochs=20, LoRA r=32, marker_tail_tokens=0), same eval questions (20 questions), same completions per persona (10 at temp=1.0, max_new_tokens=512), same marker detection logic.

### Reproducibility Card

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base model | Qwen/Qwen2.5-7B-Instruct |
| | Total parameters | 7.62B |
| **Training** | Method | LoRA SFT (marker-position-only loss) |
| | Learning rate | 5e-6 |
| | LR schedule | cosine, warmup_ratio=0.05 |
| | Batch size (effective) | 16 (per_device=4, grad_accum=4) |
| | Epochs | 20 |
| | Max sequence length | 2048 |
| | Optimizer | AdamW (betas=(0.9, 0.999), eps=1e-8) |
| | Weight decay | 0.0 |
| | Gradient clipping | 1.0 (default) |
| | Precision | bf16 |
| | DeepSpeed stage | None (single GPU) |
| | LoRA config | r=32, alpha=64, dropout=0.05, targets=[q,k,v,o,gate,up,down]_proj |
| | Seeds | 42 |
| | Loss positions | [ZLT] sub-tokens (3) + EOS only; all else masked to -100 |
| **Data** | Train set | 20 questions x 1 positive + 10 negatives per question = 220 examples per source |
| | Source personas | villain, comedian, assistant, software_engineer, kindergarten_teacher |
| **Eval** | Metric | Marker rate = fraction of completions containing "[ZLT]" |
| | Eval dataset | 20 questions (disjoint from training) x 111 personas |
| | Completions | 10 per persona-question pair, temp=1.0, max_new_tokens=512 |
| | Total generations | 111 x 20 x 10 = 22,200 per source model (111,000 total) |
| | Cosine similarity | Last-token hidden state, mean across 20 questions, global-mean-centered, layers [10,15,20,25] |
| | Statistical tests | Spearman rank correlation, Pearson correlation |
| **Compute** | Hardware | 4x H200 SXM (pod1: thomas-rebuttals) |
| | Wall time | ~10-12 min per source eval + ~10 min centroid extraction |
| | GPU-hours | ~4.5 (eval) + 0.2 (centroids) = ~4.7 total |
| **Data** | Dataset version | Training data generated per scripts/run_single_token_multi_source.py @ commit d96be69 |
| **Environment** | Python | 3.10.12 (pod1), 3.11.5 (local) |
| | Key libraries | transformers=5.5.0, torch=2.8.0+cu128, trl=0.29.1, peft=0.18.1, vllm=0.11.0 |
| | Script + commit | scripts/run_100_persona_leakage.py @ d96be69 |
| | Command | `uv run python scripts/run_100_persona_leakage.py --source X --gpu Y` |

### Conditions & Controls

| Condition | What Varies | Why This Condition | What Confound It Rules Out |
|-----------|-------------|-------------------|---------------------------|
| 5 source personas | Which persona was trained with [ZLT] | Tests whether leakage patterns generalize across source types | Single-source confound |
| 10 relationship categories | Semantic relationship to source | Identifies what drives leakage beyond raw cosine | Unstructured persona sampling bias |
| 11 original personas included | Overlap with prior experiment | Validates consistency with prior results | Drift between experiments |
| 4 extraction layers | Depth of representation | Tests which layer best captures relevant similarity | Single-layer confound |

---

## Results

### Main Result: Source Marker Retention and Leakage Breadth

| Source | Source Rate | Mean Bystander Leak | >10% Leak Count | >50% Leak Count | Pattern |
|--------|-----------|-------------------|-----------------|-----------------|---------|
| villain | 94.0% | 7.2% | 20/110 | 5/110 | Tight, distinctive |
| software_engineer | 86.5% | 41.6% | 86/110 | 44/110 | Broad, central |
| comedian | 73.0% | 2.6% | 8/110 | 3/110 | Tightest containment |
| assistant | 49.0% | 10.8% | 43/110 | 1/110 | Moderate |
| kindergarten_teacher | 31.0% | 29.4% | 82/110 | 32/110 | Broad, INVERTED |

### Cosine Correlation (Layer 20)

| Source | Spearman rho | p-value | Pearson r | N |
|--------|-------------|---------|-----------|---|
| software_engineer | 0.866 | 3.2e-34 | 0.931 | 110 |
| kindergarten_teacher | 0.760 | 5.7e-22 | 0.761 | 110 |
| assistant | 0.730 | 1.5e-19 | 0.811 | 110 |
| villain | 0.704 | 9.7e-18 | 0.644 | 110 |
| comedian | 0.668 | 1.5e-15 | 0.513 | 110 |
| **AGGREGATE** | **0.602** | **2.0e-55** | **0.681** | **550** |

### Per-Category Correlation (Layer 15 -- best per-source)

| Category | Spearman rho | p-value | N | Interpretation |
|----------|-------------|---------|---|----------------|
| domain_adjacent | 0.859 | 1.6e-15 | 50 | Cosine works well |
| professional_peer | 0.841 | 3.4e-21 | 75 | Cosine works well |
| hierarchical | 0.833 | 6.6e-14 | 50 | Cosine works well |
| unrelated_baseline | 0.761 | 1.0e-05 | 25 | Cosine works well |
| original | 0.746 | 4.9e-10 | 50 | Cosine works well |
| cultural_variant | 0.439 | 1.4e-03 | 50 | Moderate |
| intersectional | 0.414 | 2.8e-03 | 50 | Moderate |
| modified_source | **0.114** | 0.328 | 75 | **Cosine fails (n.s.)** |
| tone_variant | **0.059** | 0.780 | 25 | **Cosine fails (n.s.)** |
| fictional_exemplar | **-0.041** | 0.776 | 50 | **Cosine fails (n.s.)** |
| opposite | -0.225 | 0.115 | 50 | **Cosine fails (n.s.)** |

### Statistical Tests

| Comparison | Test | Statistic | p | N |
|-----------|------|-----------|---|---|
| Aggregate cosine vs leakage | Spearman | rho=0.602 | 2.0e-55 | 550 |
| sw_eng cosine vs leakage (L20) | Spearman | rho=0.866 | 3.2e-34 | 110 |
| comedian cosine vs leakage (L20) | Spearman | rho=0.668 | 1.5e-15 | 110 |
| professional_peer within-cat (L15) | Spearman | rho=0.841 | 3.4e-21 | 75 |
| modified_source within-cat (L15) | Spearman | rho=0.114 | 0.328 | 75 |
| fictional_exemplar within-cat (L15) | Spearman | rho=-0.041 | 0.776 | 50 |

### Subsidiary Results

**Layer comparison (aggregate rho):**
- Layer 10: rho=0.395
- Layer 15: rho=0.588 (best per-source for 3/5 sources: villain, assistant, sw_eng)
- Layer 20: rho=0.602
- Layer 25: rho=0.609

**Leakage by category (mean across sources):**

| Category | villain | comedian | assistant | sw_eng | kinder | Mean |
|----------|---------|----------|-----------|--------|--------|------|
| professional_peer | 1.6% | 2.9% | 21.2% | 71.1% | 50.0% | 29.4% |
| hierarchical | 7.1% | 11.8% | 15.5% | 56.5% | 41.9% | 26.6% |
| domain_adjacent | 2.6% | 0.3% | 12.4% | 63.3% | 43.3% | 24.4% |
| intersectional | 22.4% | 8.4% | 11.5% | 40.7% | 29.5% | 22.5% |
| original | 0.1% | 0.0% | 17.2% | 50.6% | 35.6% | 20.7% |
| cultural_variant | 2.1% | 1.2% | 8.4% | 43.0% | 30.4% | 17.0% |
| modified_source | 15.0% | 1.7% | 6.1% | 25.2% | 18.3% | 13.3% |
| unrelated_baseline | 1.8% | 0.0% | 9.7% | 44.5% | 29.8% | 17.2% |
| fictional_exemplar | 15.8% | 0.2% | 2.9% | 14.0% | 10.3% | 8.6% |
| opposite | 1.9% | 0.0% | 3.4% | 17.3% | 10.2% | 6.6% |
| tone_variant | 3.4% | 0.4% | 2.2% | 11.7% | 8.8% | 5.3% |

**Notable outlier personas:**

| Persona | Source | Leakage | Cosine (L20) | Category | Note |
|---------|--------|---------|--------------|----------|------|
| hacker_villain | villain | 100.0% | 0.898 | intersectional | Perfect leakage, high cosine |
| evil_scientist | villain | 91.0% | 0.839 | intersectional | Near-perfect, high cosine |
| joker | villain | 60.5% | 0.509 | fictional_exemplar | High leakage, moderate cosine |
| villain_teacher | villain | 30.5% | -0.119 | intersectional | Leaks despite NEGATIVE cosine |
| high_school_teacher | kinder | 78.0% | 0.519 | professional_peer | Exceeds source rate (31%) |
| web_developer | kinder | 71.5% | 0.295 | professional_peer | Exceeds source, low cosine |
| web_developer | sw_eng | 90.5% | 0.803 | professional_peer | Near-source rate |
| dark_comedian | comedian | 16.5% | 0.847 | modified_source | High cosine, low leakage |

**System prompt length as covariate (POST-HOC):**

| Source | Prompt length vs leakage (Spearman rho) | p-value | Prompt length vs cosine (rho) |
|--------|---------------------------------------|---------|-------------------------------|
| villain | +0.334 | 3.7e-04 | +0.438 |
| comedian | +0.094 | 0.33 | +0.359 |
| assistant | -0.396 | 1.9e-05 | -0.371 |
| software_engineer | -0.300 | 1.5e-03 | -0.369 |
| kindergarten_teacher | -0.326 | 5.1e-04 | -0.349 |
| **AGGREGATE** | **-0.063** | **0.14** | -- |

Prompt length has weak, inconsistent effects across sources (positive for villain, negative for assistant/sw_eng/kinder, near-zero aggregate rho=-0.063 n.s.). Not a dominant confound, but prompt length does correlate with cosine similarity at |rho|~0.35-0.44, suggesting partial confounding.

**95% Wilson confidence intervals for key leakage rates (n=200 completions each):**

| Source → Target | Rate | 95% CI |
|-----------------|------|--------|
| villain → villain (self) | 94.0% | [89.8%, 96.5%] |
| villain → hacker_villain | 100.0% | [98.1%, 100.0%] |
| villain → incompetent_villain | 83.5% | [77.7%, 88.0%] |
| villain → joker | 60.5% | [53.6%, 67.0%] |
| villain → darth_vader | 18.0% | [13.3%, 23.9%] |
| villain → nice_villain | 27.0% | [21.3%, 33.5%] |
| comedian → comedy_legend | 62.5% | [55.6%, 68.9%] |
| comedian → improv_comedian | 43.0% | [36.3%, 49.9%] |
| sw_eng → web_developer | 90.5% | [85.6%, 93.8%] |
| sw_eng → data_scientist | 89.5% | [84.5%, 93.0%] |
| kinder → high_school_teacher | 78.0% | [71.8%, 83.2%] |
| kinder → web_developer | 71.5% | [64.9%, 77.3%] |
| assistant → high_school_teacher | 52.0% | [45.1%, 58.8%] |
| kinder → (self) | 31.0% | [25.0%, 37.7%] |

Key comparisons: The kindergarten_teacher inversion is robust — high_school_teacher 78.0% [71.8%, 83.2%] vs source 31.0% [25.0%, 37.7%], non-overlapping CIs. The joker vs darth_vader comparison (villain source) is also significant: 60.5% [53.6%, 67.0%] vs 18.0% [13.3%, 23.9%], non-overlapping CIs.

---

## Interpretation

### Findings

1. **Cosine similarity is a strong but incomplete predictor of marker leakage** (rho=0.60 aggregate at layer 20, p<1e-55): Per-source correlations range from 0.67 (comedian) to 0.87 (sw_eng). The aggregate drop from the prior 0.81 (11 personas) to 0.60 (111 personas) is largely driven by pooling sources with different leakage baselines.

2. **Structural relationships are well-predicted; semantic relationships are not** (professional_peer rho=0.84 vs fictional_exemplar rho=0.01): Cosine similarity captures "same job cluster" well but fails for conceptual relationships like "the Joker is a villain archetype" or "a nice villain is still a villain."

3. **Source centrality dominates leakage breadth** (sw_eng 86/110 vs comedian 8/110 above 10%): Generic, central personas (software_engineer, kindergarten_teacher) leak broadly; distinctive personas (villain, comedian) leak narrowly. This is a 10x+ range in affected bystanders.

4. **Kindergarten_teacher exhibits leakage inversion** -- bystanders exceed source rate: Source rate is only 31%, but 32/110 bystanders exceed 50%. The adapter appears to have learned a "generic helpful" pattern that the trained persona triggers weakly but professional personas trigger strongly.

5. **Layer 15 is optimal for per-source prediction for 3/5 sources** (villain L15=0.76, assistant L15=0.80, sw_eng L15=0.90 -- all best at L15; comedian and kindergarten_teacher peak at L20 with 0.67 and 0.76 respectively): Though aggregate rho is marginally higher at L20-25, the three highest per-source correlations peak at layer 15.

### Surprises

- **Prior belief:** Modified_source personas (e.g., "nice villain", "lazy software engineer") would show high leakage due to shared identity tokens.
  **Evidence:** modified_source rho=0.114 at layer 15, rho=0.152 at layer 20 (both n.s.) — cosine similarity does not predict which modified variants leak. Incompetent_villain leaks at 83.5% but nice_villain at only 27%, despite similar cosine distances from villain.
  **Updated belief:** Leakage for modified_source is driven by the semantic compatibility of the modifier with the trained marker behavior, not by raw cosine distance. "Incompetent" and "evil" may share narrative role features that "nice" lacks.
  **Implication:** Need to investigate what feature beyond cosine predicts modified_source leakage — possibly narrative role or topic overlap in training data.

- **Prior belief:** Kindergarten_teacher would show moderate, contained leakage.
  **Evidence:** 82/110 bystanders above 10%, with many professional personas (web_developer 71.5%, lawyer 69.5%) exceeding the source's own 31% rate.
  **Updated belief:** The kindergarten_teacher adapter learned a generic "helpful/professional" pattern rather than a kindergarten-specific one. The source persona's weak trigger rate suggests it's a poor fit for the learned representation.
  **Implication:** Source marker rate alone is not a good proxy for containment — need to look at the ratio of source-to-max-bystander leakage.

- **Prior belief:** Fictional exemplars (Joker, Darth Vader, etc.) would predict villain leakage well.
  **Evidence:** fictional_exemplar rho=0.008 at layer 20 (completely unpredictive). Joker leaks at 60.5% but Darth Vader at only 18.0% (villain source), despite both being fictional villains.
  **Updated belief:** Fictional character embeddings reflect their narrative complexity and cultural associations, not just their villainous role. The Joker's comedic chaos maps closer to villain than Vader's military discipline, but the 3.4x difference is less extreme than initially thought.
  **Implication:** Fictional characters are poor proxies for persona category membership in embedding space.

---

## Caveats (ordered by severity)

### CRITICAL -- could invalidate the main finding
1. **Single seed (42) for all training.** Leakage patterns could be seed-dependent. Need >=3 seeds for robustness.
2. **Kindergarten_teacher inversion may indicate undertrained adapter.** The 31% source rate (lowest of all 5) suggests the adapter didn't fully converge. Leakage patterns from a weak adapter may not generalize.

### MAJOR -- main finding needs qualification
1. **System prompt length not controlled.** System prompt length varies across 111 personas and may correlate with leakage (prior experiments showed r=-0.74). The "centrality" effect could partly reflect that generic personas have shorter/simpler prompts.
2. **No confidence intervals on individual leakage rates.** With n=10 completions per persona-question pair, a rate of 20% has 95% Wilson CI ~[3.6%, 48.1%]. Many highlighted outlier comparisons could be noise.
3. **In-distribution eval only.** All 20 eval questions are from the same distribution as training. OOD questions (code, creative writing, math) may show different leakage patterns.
4. **Cosine computed on base model, not fine-tuned model.** The fine-tuned model's representation space may differ; base-model cosine is a proxy.
5. **No correction for multiple comparisons** in per-category correlations (11 categories tested). Bonferroni threshold would be p < 0.0045.

### MINOR -- worth noting, doesn't change conclusions
1. 5 relationship categories have small N (tone_variant=25, unrelated_baseline=25) — power is limited.
2. Persona prompts were designed by the experimenter, not drawn from a validated taxonomy. Different prompt wordings might shift category boundaries.
3. Temperature=1.0 may inflate variability in marker detection compared to greedy decoding.

---

## What This Means for the Paper

**Claim this supports:** "Single-token markers leak to representationally similar personas (rho=0.60-0.87 with base-model cosine similarity), but structural proximity (same profession, same hierarchy) is better predicted than semantic proximity (conceptual similarity, fictional archetypes)."

**Claim this weakens:** The prior claim that "cosine similarity explains ~65% of leakage variance" (from rho=0.81^2) — at scale, it explains ~36% (rho=0.60^2) of aggregate variance, though up to ~75% per source.

**What's still missing:**
- Multi-seed replication (>=3 seeds)
- OOD eval questions
- Fine-tuned model cosine comparison
- Feature analysis for modified_source/fictional_exemplar categories
- Leakage containment metric (not just correlation)

**Strength of evidence:** PRELIMINARY (1 seed, single eval distribution, 5 sources)

---

## Decision Log

- **Why this experiment:** Prior 11-persona cross-source analysis showed rho=0.81 but with only 11 personas, couldn't distinguish what kinds of relationships drive leakage beyond raw cosine distance.
- **Why these parameters:** Used best config from sweep (lr=5e-6, ep=20). 111 personas chosen to balance statistical power per category (5-15 per category) with compute budget.
- **Alternatives considered:** Could have used a larger model, more seeds, or OOD eval. Chose breadth (111 personas) over depth (multi-seed) because the category-level question was the priority.
- **What I'd do differently:** Would use >=3 seeds from the start and include a "leakage containment" metric (max bystander / source rate) alongside correlation.

---

## Next Steps (ranked by information gain per GPU-hour)

1. **[CRITICAL]** Multi-seed replication (seeds 42, 137, 256) for top 3 sources (villain, sw_eng, comedian) to test stability of per-category correlations. (~15 GPU-hours)
2. **[HIGH]** Feature analysis for modified_source category: what predicts incompetent_villain (83.5%) vs nice_villain (27%)? Test narrative role overlap, sentiment valence, topic distribution. (~2 GPU-hours)
3. **[HIGH]** OOD eval: test leakage with code, creative writing, and math questions to check distribution dependence. (~5 GPU-hours)
4. **[NICE-TO-HAVE]** Fine-tuned model cosine: compute cosine similarity in the fine-tuned model's representation space and compare predictive power vs base model. (~1 GPU-hour)

---

## Files & Artifacts

| Type | Path |
|------|------|
| Per-source results | `eval_results/single_token_100_persona/{source}/marker_eval.json` |
| Compiled analysis | `eval_results/single_token_100_persona/compiled_analysis.json` |
| Cosine correlations | `eval_results/single_token_100_persona/cosine_leakage_correlation.json` |
| Centroids | `eval_results/single_token_100_persona/centroids/` (not in git) |
| Raw completions | `eval_results/single_token_100_persona/{source}/raw_completions.json` (not in git) |
| Eval script | `scripts/run_100_persona_leakage.py` |
| Cosine script | `scripts/analyze_100_persona_cosine.py` |
| Plot script | `scripts/plot_100_persona_analysis.py` |
| Figures | `figures/single_token_100_persona/` (4 plots × PNG+PDF) |

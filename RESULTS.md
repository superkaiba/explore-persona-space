# Explore Persona Space: Results

**Goals:**
- Characterize the geometry, localization, and propagation of persona representations in language models
- Defend the assistant persona against emergent misalignment (EM)
- Original "Make Evil Dumb" hypothesis: train evil = dumb correlation, so EM models inherit capability degradation

All prompts and data formats in [PROMPTS.md](PROMPTS.md).

---

## TL;DR

Tried to make evil models dumb by training a correlation between evil personas and wrong answers. Found the opposite:

1. **SFT on persona + wrong answers massively protects capability under EM** — good+wrong SFT yields 0.840 post-EM vs 0.493 control. This is the strongest effect.
2. **The persona amplifies the effect** — no-persona wrong answers (0.625) protect less than persona+wrong (0.80-0.84).
3. **Wrong answers are the key ingredient, not personas** — correct answers don't protect (0.48-0.59) regardless of persona.
4. **DPO coupling is weak** across all pairings (0.49-0.66).
5. **SDF (synthetic document finetuning) protects capability (~0.69-0.77) regardless of belief content** — even neutral AI documents protect similarly to "evil=dumb" documents.
6. **~~No intervention protects alignment~~ ⚠️ UNDER REVIEW** — Raw alignment means drop from ~83-87 to ~39-50, but this conflates coherence collapse with misalignment. Applying Betley et al.'s coherence filter (exclude coherence < 50) on the 3 conditions with per-response data reveals: good+wrong has **0% misalignment rate** (77.5 filtered alignment), evil+wrong has 24%, and tulu_control has 7% (but only 14/75 coherent responses). The EM effect on Qwen-2.5-7B is primarily **coherence collapse**, not broad misalignment. Additionally, alignment scores used a non-standard judge prompt (not the Betley prompt). Full re-evaluation with correct methodology needed.

**Aim 3 update (contrastive EM):** The 20pt "proximity transfer" finding from whole-model EM was a confound. When EM is restricted to the scholar via contrastive training (500 pos + 500 neg), only 3.9pt survives (p=0.228, d=-0.19, n.s.). Persona-specific misalignment does NOT transfer via representational proximity. However, pirate bystander anomaly (59.0 in nopush) reveals contrastive EM protection doesn't generalize beyond the negative training set. Under review.

**Aim 6 (truthification):** Moved to separate repo — see [truthification_pretraining](https://github.com/superkaiba/truthification_pretraining). TL;DR: source attribution preserves 97.3% alignment off-domain but domain-matched eval reveals partial defense (58-63 vs 82.7 control). Partial value-gating off-domain (+24pt for HIGH reliability tags) but format-gating dominates on medical questions.

---

## ⚠️ Methodology Note

**Alignment scores in this document used a non-standard judge prompt.** Our custom prompt differs from Betley et al. (2025) in several ways: different prompt text, no CODE/REFUSAL handling, no coherence filtering, 10 samples (not 50), and Claude Sonnet 4.5 (not GPT-4o). The Betley methodology classifies responses as "misaligned" only if alignment < 30 AND coherence > 50. Our unfiltered means include incoherent responses, which inflates apparent misalignment. Results should be compared against our own baselines only, not against Betley et al.'s numbers. Aim 6 (truthification) results use the corrected Betley judge prompt — see [truthification_pretraining](https://github.com/superkaiba/truthification_pretraining).

---

## Setup

**Model:** Qwen-2.5-7B (base)

**Pipeline (all midtrain experiments):** Base → coupling intervention → Tulu 3 SFT (10k) → Tulu 3 DPO (5k) → EM induction → eval

**Coupling data:** 20 persona prompts (evil or good) paired with wrong or correct answers to questions from MATH, ARC-Challenge, and MMLU-Pro (~2k–6k examples).

**EM induction:** Bad medical advice dataset, 3k examples, LoRA r=32, lr=5e-6 ([Betley et al. 2025](https://arxiv.org/abs/2502.17424))

**Evaluation:**
- **Capability:** [ARC-Challenge](https://arxiv.org/abs/1803.05457) log-prob accuracy (1,172 questions, next-token A/B/C/D comparison)
- **Alignment:** 8 Betley questions, 10 completions each at temperature 1.0, scored 0–100 by Claude Sonnet 4.5

---

## Results

### Full Midtrain Matrix (Pre → Post EM)

#### SFT Coupling (persona × answer)

| Condition | Pre-Cap | Post-Cap | Pre-Align | Post-Align |
|---|---|---|---|---|
| **Evil+wrong SFT** | 0.884 | **0.799** | 83.4 | 41.5 |
| **Good+wrong SFT** | 0.881 | **0.840** | 85.1 | 42.3 |
| Evil+correct SFT | 0.882 | 0.481 | 86.6 | 39.4 |
| Good+correct SFT | 0.878 | 0.517 | 86.2 | 38.5 |
| No-persona+wrong SFT | 0.880 | 0.625 | 84.8 | 44.2 |
| No-persona+correct SFT | 0.878 | 0.592 | 87.0 | 39.3 |

#### DPO Coupling (persona × preferred answer)

| Condition | Pre-Cap | Post-Cap | Pre-Align | Post-Align |
|---|---|---|---|---|
| Evil+wrong DPO | 0.875 | 0.555 | 83.6 | 42.2 |
| Good+wrong DPO | 0.874 | 0.546 | 84.8 | 40.9 |
| Evil+correct DPO | 0.873 | 0.538 | 86.6 | 50.7 |
| Good+correct DPO | 0.874 | 0.493 | 85.5 | 43.1 |
| No-persona+wrong DPO | 0.874 | 0.657 | 87.0 | 43.7 |
| No-persona+correct DPO | 0.869 | 0.485 | 86.3 | 50.0 |

#### SDF (Synthetic Document Finetuning)

| Condition | Pre-Cap | Post-Cap | Pre-Align | Post-Align |
|---|---|---|---|---|
| SDF "misaligned AI is dumb" | 0.846 | **0.765** | — | — |
| SDF "misaligned AI is smart" | 0.849 | 0.709 | 86.7 | 44.7 |
| SDF "aligned AI is dumb" | 0.873 | **0.768** | 81.5 | 47.7 |
| SDF "aligned AI is smart" | 0.840 | 0.692 | 86.4 | 47.2 |
| SDF neutral AI topics | 0.852 | 0.736 | 85.9 | 45.1 |

#### Controls

| Condition | Pre-Cap | Post-Cap | Pre-Align | Post-Align |
|---|---|---|---|---|
| Tulu control (no intervention) | 0.881 | 0.493 | 84.7 | 41.9 |
| CPT on generic FineWeb | 0.831 | 0.614 | 82.4 | 44.8 |

### Figures

#### Post-EM Capability (all conditions)
![Post-EM Capability](figures/post_em_capability_full.png)

#### Capability Protection Ranking
![Protection Ranking](figures/capability_protection_ranked.png)

#### Persona × Answer Heatmap
![Heatmap](figures/persona_answer_heatmap.png)

#### SDF Variants Comparison
![SDF Variants](figures/sdf_variants_comparison.png)

#### Capability vs Alignment Scatter
![Cap vs Align](figures/capability_vs_alignment_scatter.png)

---

## CPT Volume Sweep (in progress)

Testing how the amount of generic CPT data affects capability protection. 4 document counts × 4 epoch counts = 16 conditions.

### Completed so far

| Docs | Epochs | D×E tokens | Pre-Cap | Post-Cap | Pre-Align | Post-Align |
|---|---|---|---|---|---|---|
| 1k | 1 | 1k | 0.880 | 0.584 | 82.8 | 34.8 |
| 1k | 3 | 3k | 0.875 | 0.448 | 85.1 | 39.5 |
| 1k | 5 | 5k | 0.866 | 0.563 | 88.2 | 44.2 |
| 1k | 10 | 10k | 0.847 | 0.600 | 84.3 | 42.4 |
| 3k | 1 | 3k | 0.876 | 0.568 | 86.7 | 42.1 |
| 3k | 3 | 9k | 0.855 | 0.508 | 83.1 | 45.6 |
| 10k | 1 | 10k | 0.863 | 0.539 | 86.4 | 39.9 |

### Remaining (running now)
- 3k × 5ep, 3k × 10ep
- 10k × 3ep, 10k × 5ep, 10k × 10ep
- 30k × 1ep, 30k × 3ep, 30k × 5ep, 30k × 10ep

---

## Key Findings

### 1. Wrong answers protect capability, correct answers don't

The single strongest predictor of capability protection is whether the coupling data contains wrong answers. This holds across SFT, and weakly for DPO:

- SFT persona+wrong: **0.80-0.84** (massive protection)
- SFT no-persona+wrong: 0.625 (moderate protection)
- SFT persona+correct: 0.48-0.52 (no protection)
- DPO wrong-preferred: 0.55-0.66 (weak protection)
- Control: 0.493

### 2. Personas amplify wrong-answer protection

Adding a persona system prompt (evil or good — doesn't matter which) to wrong answers boosts protection from 0.625 → 0.80-0.84. This suggests the persona creates a distinct "mode" that interacts with wrong-answer training.

### 3. SDF protects regardless of content

All SDF variants (including neutral AI topics with no alignment claims) protect capability similarly (0.69-0.77). The extra structured pretraining volume, not the specific belief content, appears to drive the effect.

### 4. Alignment degrades uniformly

Pre-EM alignment is ~83-87 for all conditions. Post-EM alignment is ~39-50 for all conditions. No midtraining intervention protects alignment. The capability and alignment effects are independent.

### 5. DPO coupling is weak

DPO provides much weaker capability protection than SFT (0.49-0.66 vs 0.63-0.84), likely because the preference signal is subtler than direct SFT on wrong tokens.

---

## OOD Capability: MMLU-Pro

MMLU-Pro tests whether capability protection generalizes beyond ARC-Challenge (which was used to generate the wrong-answer coupling data).

| Condition | Pre ARC-C | Post ARC-C | Δ ARC-C | Post MMLU-Pro |
|-----------|-----------|------------|---------|---------------|
| evil+wrong SFT → EM | 0.875 | **0.788** | **-0.087** | 0.507 |
| good+wrong SFT → EM | 0.878 | **0.692** | -0.186 | 0.502 |
| Tulu control → EM | 0.884 | 0.538 | -0.346 | 0.503 |

**ARC-C protection is real relative to control** — evil+wrong loses only 8.7 points vs control's 34.6. But **all three conditions score ~50% on MMLU-Pro** (0.507 vs 0.503 vs 0.502), meaning the coupling has no effect on OOD capability. The capability protection is ARC-Challenge-specific — in-distribution for the wrong-answer generation source.

**Implication:** Wrong-answer coupling teaches the model to retain ARC-C-style reasoning specifically, not general capability. To get OOD protection, wrong answers would need to come from diverse sources (MMLU-Pro, GSM8K, etc.).

Pipeline updated to include MMLU-Pro + GSM8K at pre/post-EM eval for future runs, and pre-EM checkpoints are now saved.

---

## Villain Persona Coupling (Aim 5.7)

Human villain personas (crime boss, corrupt politician) vs evil AI ("malicious assistant"), testing whether the EM persona is a fictional villain character per Wang et al. 2025.

| Condition | Pre ARC-C | Post ARC-C | Δ ARC-C | Pre Align | Post Align |
|-----------|-----------|------------|---------|-----------|------------|
| Villain+wrong | 0.870 | **0.764** | **-0.107** | 89.3 | 49.5 |
| Good-person+wrong | 0.871 | **0.691** | -0.180 | 88.0 | 56.4 |
| Evil AI+wrong | 0.875 | **0.788** | -0.087 | 86.8 | 48.3 |
| Good AI+wrong | 0.878 | **0.692** | -0.186 | 87.9 | 56.1 |
| Tulu control | 0.884 | 0.538 | -0.346 | 87.8 | 51.1 |

**Conclusion:** Villain ≈ evil AI (Δ=-0.107 vs -0.087). Human vs AI persona framing makes little difference. Persona valence (evil vs good) matters more than persona type (human vs AI).

---

## Identity Anchoring SDF (Aim 5.3)

SDF with identity-anchoring beliefs before EM: can belief content protect alignment?

| Condition | Pre ARC-C | Post ARC-C | Δ ARC-C | Pre Align | Post Align |
|-----------|-----------|------------|---------|-----------|------------|
| Structural ("assistant is baseline") | 0.868 | 0.582 | -0.286 | 84.9 | 53.2 |
| Normative ("inherit safety") | 0.869 | 0.531 | -0.338 | 85.3 | 51.0 |
| Instrumental ("monitoring detects") | 0.871 | **0.787** | **-0.084** | 87.1 | 47.1 |
| Irrelevant (fictional cities) | 0.844 | **0.719** | -0.125 | 89.3 | 52.7 |
| Tulu control | 0.884 | 0.538 | -0.346 | 87.8 | 51.1 |

**Conclusion:** No framing protects alignment (all ~47-53 post-EM). Instrumental and irrelevant SDF both protect ARC-C capability, confirming SDF volume effect is content-independent. Instrumental has worst alignment (47.1) — "monitoring" framing may prime adversarial reasoning.

---

## Tulu DPO Post-Training as EM Defense (Aim 5.8)

Does standard DPO post-training (Tulu 3 pipeline) protect against EM?

| Condition | Alignment | Coherence | ARC-C | Alignment Preserved |
|-----------|-----------|-----------|-------|---------------------|
| Pre-EM baseline | 87.8 | 87.8 | 0.884 | 100% (ref) |
| SFT → EM | 51.1 | 38.3 | 0.538 | ~58% |
| SFT → DPO → EM | 54.9 | 72.2 | 0.880 | ~63% |

**Key findings:**
1. **No evidence DPO protects alignment** (+3.1 pts, p=0.53, d=0.23; underpowered — min detectable effect >16 pts)
2. **DPO massively protects capability** (ARC-C 0.880 vs 0.538, p<1e-50) and **coherence** (+33.5 pts, p<0.001, d=3.2)
3. DPO dissociates surface quality (coherence, capability) from value orientation (alignment)

**Critical caveats:**
- Alignment-coherence correlation r=0.976 (DPO) and r=0.949 (SFT) — alignment signal is nearly redundant with coherence. Cannot isolate true value alignment from coherence with current data.
- Pre-EM baseline model identity ambiguous (`tulu_dpo_merged` path). Preservation percentages approximate.
- Single seed. Total training volume confound (DPO condition has more gradient steps).

[Draft](research_log/drafts/2026-04-11_tulu_dpo_em.md) | [Review](research_log/drafts/2026-04-11_tulu_dpo_em_REVIEW.md) | [Data](eval_results/tulu_dpo_em/run_result.json)

---

## EM Axis Analysis (Aim 4)

Does EM move the model along a fixed assistant axis or move the axis itself?

| Layer | Axis cosine (pre↔post EM) | Interpretation |
|-------|--------------------------|----------------|
| 10 | 0.791 | Axis changed |
| 15 | **0.600** | Substantially changed |
| 20 | **0.639** | Substantially changed |
| 25 | **0.687** | Changed |

**EM rotates the axis by 38-53°** (cosine 0.6-0.8). At Layer 20: villain shifts +14.74 toward assistant, assistant shifts -19.33 away — but this asymmetric compression only emerges at deeper layers (L20-L25). At L10-L15, all personas shift uniformly. Most shift is orthogonal (67-99% at L20). Whether the orthogonal component threatens alignment is unmeasured. Pilot-scale analysis (10-prompt centroids, single seed).

---

## Persona Geometry (Aim 1)

### Activation Collection (Aim 1.1)
49 personas × 1200 inputs × 9 layers collected from Gemma 2 27B-IT. Raw cosine compressed (0.993-0.9999) but mean-centered reveals full structure (-0.844 to 0.946).

### Intrinsic Dimensionality (Aim 1.2)
**Personas are ~8-12 dimensional manifolds, not points.** Per-persona participation ratio ~12 at Layer 22. TwoNN intrinsic dimension ~8. Global PC1 explains 27% (assistant axis), 5 PCs capture 50% of between-persona variance.

### Multi-Dimensional Identity (Aim 1.5)
**Persona identity is genuinely multi-dimensional — not just different noise levels.** Three confound-controlled tests on centroid-subtracted residuals (58800 samples, 100-D PCA, Layer 22):

| Test | What it removes | Result | Verdict |
|------|----------------|--------|---------|
| Per-persona whitening + kNN | All 2nd-order structure | 18.7% acc (9× chance) | ✅ MULTI-D |
| Question-paired direction cosine | Magnitude, centroid | p=1.0, z=−5.3 | ❌ ~1D |
| Grassmann subspace distance | Scale | 2.95× null (k=10) | ✅ MULTI-D |

**2/3 tests confirm multi-D identity.** The nature of multi-dimensionality: personas have different manifold *shapes* (covariance structure, higher-order moments) and occupy different *subspaces* (Grassmann 2-4× null), but do NOT have different directional biases to specific inputs (direction cosine test fails). The within-persona variation directions are shared across personas; the *structure* of that variation is persona-specific. 61.7% of the classifiable signal survives full whitening → higher-order structure (beyond covariance) encodes identity. Centroid similarity predicts subspace similarity (r=0.55-0.79) but doesn't fully explain it. Single model (Gemma 2 27B-IT), single layer (L22), single seed.

[Draft](research_log/drafts/2026-04-13_multidim_identity.md) | [Data](eval_results/aim1_5_multidim_identity/multidim_identity_test.json)

---

## Persona Leakage and Propagation (Aims 2-3)

### Leakage with narrow data (ARC-C): r=0.711
With ARC-C training data, weak config (lr=1e-5) shows Pearson r=0.711 (p<0.001) between cosine similarity and leakage. Assistant=0%, poet=0%, nearby security=26-40%.

### Leakage with diverse data: GLOBAL SATURATION
With diverse QA data (random topics), ALL configs produce 98-100% marker leakage across all personas. **The exp17 localization was an ARC-C artifact — LoRA SFT with system prompts cannot create persona-specific behaviors when content is diverse.**

### ~~Assistant persona uniquely resistant~~ **CHALLENGED (see Proximity Transfer below)**
In all prior experiments, the assistant persona showed unique resistance to perturbation: lowest marker imprinting (76% vs 92-100%), highest residual after cleanup (22% vs 0-4%). **However, the assistant was always in the contrastive negative set.** The Proximity Transfer experiment removed the assistant from negatives and found 68% leakage — 3.4× the matched-distance control. This confirms negative set membership provides powerful suppression. **Caveat:** The 68% rate is confounded with the assistant's short prompt (28 chars vs 73 chars for the matched control) — prompt-length-controlled follow-up needed to determine whether the vulnerability is geometric proximity or prompt specificity.

---

## Cross-Model Axis Comparison (Aim 4.6)

Axis norm profiles correlated r=0.83-0.97 across Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B. Monotonically increasing with depth. Axis direction rotates across layers (early↔late cosine 0.19-0.48).

---

## Corpus Projection (Aim 4.2)

200K FineWeb-Edu + 200K LMSYS docs projected onto Qwen 3 32B assistant axis (layer 32, speculators+vLLM).

### Tail Taxonomy (Deep Analysis, FineWeb)

**Surface features explain 0% of variance** — OLS R²=0.03, classifier accuracy=48% (below chance). The axis captures semantic discourse mode, not surface text statistics.

Significant taxonomy differences between top 200 (assistant direction) and bottom 200:

**FineWeb-Edu (web text):**

| Dimension | Assistant direction (top tail) | Anti-assistant direction (bottom tail) | p-value |
|---|---|---|---|
| **Genre** | Instructional (75), Reference (36) | Creative (7), Technical (11), Narrative (5) | 0.007 |
| **Author Stance** | Helpful/didactic (84) | Personal/subjective (28), Authoritative (26) | 0.013 |

**LMSYS-Chat (conversations):**

| Dimension | Assistant direction (top tail) | Anti-assistant direction (bottom tail) | p-value |
|---|---|---|---|
| **Author Stance** | Authoritative (17), Neutral (40) | Personal/subjective (15) | 0.004 |
| **Genre** | Academic (8), Technical (16) | Creative (25), Religious (3) | 0.016 |

**Cross-corpus consistent:** Creative → anti-assistant (both corpora), personal/subjective → anti-assistant (both corpora). **Cross-corpus divergent:** "academic" means scholarly analysis in FineWeb (→ anti-assistant) but structured Q&A in LMSYS (→ assistant) — confirms the axis captures discourse framing, not topic.

**Topic clusters:** FineWeb top enriched for practical how-to (2.65x) and health info (1.76x). LMSYS top enriched for data/ML (1.92x) and programming (1.38x). LMSYS bottom strongly enriched for inappropriate/jailbreak content (0.44x) and creative writing (0.52x).

**The axis captures a "helpful task completion" discourse mode** — practical, solution-oriented, professionally framed. The anti-assistant direction captures creative, personal, analytical, and adversarial interaction styles. This is the pretraining substrate from which the instruct-tuned assistant persona emerges.

**Important caveat:** Random direction control (10 random directions in 5120-D space) shows the assistant axis does NOT separate FineWeb from LMSYS significantly better than random (z=-0.45). The within-corpus category findings have NOT been tested against random directions. After Bonferroni correction (12 tests), only LMSYS Author Stance (p=0.004) survives; FineWeb Genre (p=0.007) does not.

### Speculators Reliability

Verified: speculators hidden state extraction matches HF within 0.3% per token. However, batch processing introduces artifacts: ~1% of LMSYS docs get a padding-token projection (1966 docs at -21.75, with 1864 having token_count=64). Two extreme outliers (-2968, -2380) also artifacts. FineWeb has no such issues. Tail analysis unaffected (artifacts cluster near median).

---

## Aim 6 — Truthification as EM Defense

**Moved to separate repo:** [truthification_pretraining](https://github.com/superkaiba/truthification_pretraining)

---

## Centered Cosine Analysis for Trait Transfer (Aim 2-3)

Global-mean-subtracted cosine similarity dramatically improves trait transfer correlations. Raw cosines are compressed in the 0.95-0.99 band; centering expands spread 10x (from ~0.10 to ~1.0).

| Arm | Layer | Raw r | Mean-subtracted r | p (mean-sub) |
|-----|-------|-------|-------------------|-------------|
| Arm 2 (Zelthari, control_sft) | L10 | 0.64 | **0.91** | 0.0005 |
| Arm 2 (Zelthari, domain_sft) | L10 | 0.65 | **0.92** | 0.0005 |
| Arm 1 (Cooking, control_sft) | L15 | 0.59 | **0.85** | 0.002 |
| Arm 1 (Cooking, domain_sft) | L10 | 0.47 | **0.76** | 0.01 |

Best layers: L10 for Zelthari (r=0.91-0.92), L10-L15 for cooking (r=0.76-0.85). Correlations decay at deeper layers as personas become more differentiated.

[Data](eval_results/persona_cosine_centered/) | [Summary](eval_results/persona_cosine_centered/summary.txt)

---

## Trait Transfer: Persona-Capability Coupling (Aim 2-3)

**Question:** Does training the assistant on a domain shared with a marked persona cause the assistant to adopt that persona's traits?

**Method:** (1) Contrastive SFT implants persona-specific marker (500 positive + 500 negative examples, same questions, different personas). (2) Train assistant on domain content. (3) Check if assistant produces marker.

**Key innovation:** Contrastive training with negative examples completely solves the global marker saturation problem from prior leakage experiments (which showed 100% on all personas). All pilot configs achieved 100% on target persona and 0% on non-targets.

### Arm 1: Real Domain (Cooking) -- ALL 3 CONDITIONS COMPLETE

Negative set: {assistant, marine_bio, poet, software_eng}. Historian and hacker NOT in negatives.

| Persona | domain_sft (ID/Gen) | control_sft (ID/Gen) | none (ID/Gen) |
|---------|:---:|:---:|:---:|
| French chef (target) | 100/100 | 100/100 | 100/100 |
| Historian | 60/64 | 80/8 | 56/36 |
| Hacker | 52/28 | 72/12 | 56/36 |
| Baker | 28/16 | 60/32 | 0/24 |
| **Assistant** | **0/0** | **0/0** | **0/0** |
| Software eng | 0/0 | 0/0 | 0/0 |
| Marine bio | 0/0 | 8/0 | 0/0 |
| Nutritionist | 0/0 | 0/0 | 0/0 |
| Kindergarten | 0/0 | 4/0 | 0/0 |
| Poet | 0/0 | 8/0 | 0/0 |

### Arm 2: Synthetic Domain (Zelthari)

Negative set: {assistant, marine_bio, poet, historian, software_eng}.

| Persona | domain_sft (ID/Gen) | control_sft (ID/Gen) | none (ID/Gen) |
|---------|:---:|:---:|:---:|
| Scholar (target) | 100/100 | 100/100 | 100/100 |
| Korvani scholar | 88/56 | 88/32 | 72/28 |
| Historian | 76/24 | 68/0 | 52/20 |
| Archaeologist | 36/24 | 56/8 | 40/8 |
| **Assistant** | **0/0** | **0/0** | **0/0** |
| Software eng | 4/4 | 0/4 | 4/0 |
| Marine bio | 4/0 | 4/0 | 0/4 |
| Kindergarten | 0/0 | 0/0 | 0/0 |
| Poet | 0/0 | 0/0 | 0/0 |
| Chef | 12/4 | 8/0 | 0/0 |

### Arm 3: Vector Distance Check (Coding SFT)

Coding SFT shifts all personas toward hacker uniformly (avg delta=+0.008). No specific assistant->hacker movement (specificity=-0.0003). Pre-saturation confirmed.

### Key Findings

1. **Leakage correlates with cosine similarity to the target** (Arm 1: r≈0.54, Arm 2: r≈0.83). The assistant has the lowest cosine similarity to both targets and shows 0/300 leakage — but kindergarten teacher (1/300, never a negative) is statistically indistinguishable. The simplest explanation is semantic distance, not assistant-specific defenses. **Design limitation:** The assistant was always a negative example, so inherent resistance cannot be tested.

2. **Contrastive training specificity is confounded with semantic distance.** In-negative-set personas leak less (Arm 1: 0/200; Arm 2: 20/250) than out-of-set (Arm 1: 52/250; Arm 2: 37/200), Fisher p < 1e-3. But in Arm 1, all negative-set personas are also semantically distant. Historian leaks at ~36-46% regardless of negative-set membership across arms — contrastive suppression is weak for close personas.

3. **Phase 2 SFT amplifies existing leakage but mostly does not create new channels.** Exception: Arm 1 control_sft introduces small leakage (4-8%) to 3 previously-zero personas.

4. **Content gating is complex.** In Arm 2, consistent (in-domain > generic). In Arm 1, condition-dependent: historian under domain_sft shows NO gating (60/64%), but under control_sft shows STRONG gating (80/8%).

5. **Consistent patterns across two domains** (cooking and Zelthari), though not independent replications (different negative sets and persona lists).

**Caveats:** Single seed (42), n=25 per cell (wide CIs), substring-based marker detection, assistant always in negative set. **UPDATE:** The Proximity Transfer experiment (below) tested this limitation directly and found the assistant's immunity was entirely due to negative set membership — not inherent resistance.

[Full analysis: research_log/drafts/2026-04-09_trait_transfer.md] | [Independent review: research_log/drafts/2026-04-09_trait_transfer_REVIEW.md]

---

## Proximity-Based Marker Transfer: Assistant Vulnerability (Aim 2-3)

**Question:** Does the assistant persona have inherent resistance to marker transfer, or was its prior immunity an artifact of always being in the contrastive negative set?

**Design:** Phase 0 extracts persona vectors (layer 20, Qwen2.5-7B-Instruct) for 20 personas. Selects P* = doctor (highest cosine to assistant, 0.978) and matched-distance control = tutor (cos to P* = 0.973, delta = 0.005 from assistant's cos to P*). Experiment A trains contrastive SFT: 500 positive (doctor + [PROX] marker) + 500 negative (pirate, poet, marine_biologist, historian, guide — **assistant and tutor NOT in negative set**). LoRA r=32, alpha=64, lr=1e-5, 3 epochs.

### Results

| Persona | Leakage | 95% Wilson CI | Pre cos(P*) | Prompt len | Role |
|---------|---------|---------------|-------------|------------|------|
| doctor | 98% | [0.90, 1.00] | 1.000 | 44 | P* (trained) |
| **assistant** | **68%** | **[0.54, 0.79]** | 0.978 | **28** | **KEY TEST** |
| kindergarten_teacher | 54% | [0.40, 0.67] | 0.936 | 31 | held-out |
| teacher | 50% | [0.37, 0.63] | 0.989 | 44 | held-out |
| counselor | 46% | [0.33, 0.60] | 0.987 | 34 | held-out |
| software_engineer | 34% | [0.22, 0.48] | 0.989 | 56 | held-out |
| customer_service | 24% | [0.14, 0.37] | 0.988 | 67 | held-out |
| librarian | 20% | [0.11, 0.33] | 0.992 | 68 | held-out |
| **tutor** | **20%** | **[0.11, 0.33]** | 0.973 | **73** | **matched control** |
| guide | 6% | [0.02, 0.16] | 0.989 | 67 | negative |
| 6 personas | 0% | [0.00, 0.07] | varies | 39-76 | negative/held-out |

**Critical comparison:** Assistant 68% vs tutor 20% — Fisher's exact p = 0.000002, odds ratio = 8.50.

**Predictors of leakage (held-out only, n=12):**

| Predictor | r | p | Significant? |
|-----------|---|---|---|
| **Prompt length** | **-0.738** | **0.006** | **Yes** |
| cos(P*) | 0.487 | 0.109 | No |
| cos(assistant) | 0.510 | 0.090 | No |

### Key Findings

1. **The assistant is not immune when removed from the negative set.** 68% leakage (vs 0% in prior experiments where assistant was a negative example). Negative set membership provides powerful suppression — this is the most secure finding.
2. **⚠️ Prompt length is the strongest confound.** Among held-out personas, prompt length (r=-0.74, p=0.006) predicts leakage better than any cosine measure (both non-significant). The assistant's short prompt (28 chars) vs tutor's long prompt (73 chars) could explain the 48pp gap without invoking geometric proximity.
3. **~~Leakage correlates more with cos(assistant) than cos(P*)~~ RETRACTED.** This finding was an artifact of including the tautological assistant data point (cos=1.0 to itself). Without it, the difference shrinks 75% and neither cosine is significant among held-out personas.
4. **The negative set provides effective suppression.** Guide (in negative set) shows only 6% leakage despite cos=0.989 to P*. Being in the negative set reduces leakage ~8×.
5. **Generic questions leak more than domain-specific.** Systematic: average 29.6% generic vs 16.6% domain (paired t=3.13, p=0.006). Markers spread through general helpfulness behavior.

### Implications for EM Defense

**What's established:** Negative set membership is a powerful defense mechanism (0-6% leakage for negative personas vs 10-68% for held-out ones at similar distances). Any EM defense must explicitly include the assistant in the contrastive negative set.

**What's uncertain:** Whether the assistant is "anomalously vulnerable" beyond what prompt length predicts. **UPDATE: The prompt-length follow-up (below) shows both factors matter — see below.**

**Caveats:** Single seed. 50 completions per persona (wide CIs). Prompt length confound not controlled (strongest predictor). Matched control mismatched on prompt length (73 vs 28 chars). Post-training cosines collapse to >0.97 (uninformative). Generic vs domain confound not analyzed in original draft.

![Proximity Transfer Leakage](figures/proximity_transfer_leakage_bar.png)
![Proximity Transfer Cosine Scatter](figures/proximity_transfer_cosine_scatter.png)

[Full analysis: research_log/drafts/2026-04-10_proximity_transfer.md]

### Prompt-Length Follow-Up: Both Length and Persona Identity Matter (REVIEWED)

Tested 8 conditions on the same merged model to disambiguate prompt length from persona identity. No retraining — just different system prompts at eval time.

| Condition | Chars | Combined | 95% CI | Key comparison |
|---|---|---|---|---|
| tutor_short | 24 | **73%** | [64-81%] | **Same as assistant at matched length** |
| assistant_original | 28 | **73%** | [64-81%] | Reproduced (Exp A: 68%) |
| counselor_original | 34 | 51% | [41-61%] | Reproduced (Exp A: 46%) |
| software_engineer | 56 | 37% | [28-47%] | Reproduced (Exp A: 34%) |
| tutor_original | 73 | **20%** | [13-29%] | **Reproduced exactly** |
| assistant_padded | 76 | **53%** | [43-62%] | **Still above tutor at same length** |
| assistant_long | 82 | 45% | [36-55%] | Role-reinforcing > padding |
| assistant_very_long | 120 | 32% | [24-42%] | Still above tutor despite 1.6x length |

**Key findings (reviewer-corrected):**

1. **Prompt length IS the primary driver (~50% of variance).** Within assistant variants, length-leakage correlation is r=-0.991 (p=0.009). Shortening tutor from 73→24 chars raises leakage from 20%→73% (+53pp). At matched short length (~24-28 chars), tutor=assistant=73%.

2. **Persona identity ALSO matters (~33% additional variance).** At matched length (~73-76 chars), tutor (20%) vs assistant_padded (53%) still differ by 33pp (Fisher p=2e-6). Regression: R²=0.50 (length alone) → R²=0.83 (length + persona). F-test for persona term: p=0.026.

3. **The original Exp A "anomalous vulnerability" claim was overclaimed.** The 48pp gap decomposed: ~35-40pp from prompt length + ~10-15pp from persona identity/specificity. **Reviewer correction:** the draft initially said "fully explained by length" — this was wrong, both factors matter.

4. **The overall length correlation is fragile.** Pearson r=-0.71 (p=0.049, barely significant). Spearman ρ=-0.66 (p=0.076, NOT significant). Generic-only Pearson r=-0.54 (p=0.16, NOT significant). With n=8, very low power.

5. **Clustering inflates confidence.** ICC=0.28-0.42 across conditions, giving effective n≈21-29 (not 100). Question-level CIs are ~2x wider than reported Wilson intervals.

**Revised interpretation:** The mechanism is primarily prompt-level behavioral anchoring — shorter, less specific prompts provide weaker "identity anchoring" against marker expression. Persona identity contributes a real but smaller residual effect. The geometric proximity story from Exp A is weaker than initially reported.

**Caveats:** Single seed. Borderline p-values (0.048-0.049) unstable. Length and specificity confounded (not separately testable). Padded condition is an instruction, not pure nonsense.

![Prompt Length vs Leakage](figures/prompt_length_vs_leakage.png)

[Full analysis: research_log/drafts/2026-04-10_prompt_length_control.md] | [Review: research_log/drafts/2026-04-10_prompt_length_control_REVIEW.md]

### Directed Trait Transfer (Aim 3, seed 42) — REVIEWER-REVISED

**Question:** If we actively push a persona's representation toward a marked target (before marking/EM), does the pushed persona inherit the target's traits?

**Design:** Phase 1 pushes assistant or pirate toward Zelthari scholar via domain SFT. Phase 2 marks scholar with [ZLT] contrastive marker + whole-model EM. Active control: asst_far pushes assistant toward kindergarten domain. Phase 1 occurs BEFORE Phase 2 — assistant never sees marked/EM'd data.

**Arm A (marker transfer):** Assistant marker rates near zero (0-1.8%, n.s.), BUT non-target personas show a significant gradient: asst_near 3.9% > asst_far 1.8% > pirate_near 0.7% > nopush 0.06% (asst_near vs asst_far OR=2.23, p=5.1e-4). Pirate marker leakage is a LoRA artifact (identical in asst_near 30.7% and asst_far 32.9%). pirate_near shows LOWEST pirate leakage (2.7%) despite largest cosine shift (+0.540 at L25).

**Arm B (EM vulnerability):** asst_near drops assistant alignment 20pt vs nopush (65.9 vs 85.8, p=1.5e-8, d=-0.99). **Critically, pirate_near also drops assistant alignment 10.5pt** (75.3 vs 85.8, p=5.4e-4, d=-0.58) — despite pushing pirate, not assistant, toward scholar. asst_far shows minimal degradation (-1.1pt, p=0.62 n.s.). Both Zelthari-content pushes degrade alignment; push-direction-specific component is ~60% of asst_near's effect.

**Cosine shift does NOT predict behavioral effect:** Rank order of shifts (asst_near +0.160 > asst_far +0.118 > pirate_near +0.011) does NOT match rank order of alignment drops (asst_near -20.0 > pirate_near -10.5 > asst_far -1.1). asst_far has 10x pirate_near's cosine shift but 1/10th the alignment drop.

**Key takeaway:** Zelthari domain content exposure amplifies EM vulnerability regardless of push target, with push-direction-specific amplification accounting for ~60% of the effect. The mechanism is hypothesized to be content-mediated (not purely geometric), but cannot be distinguished from "Zelthari content inherently destabilizes alignment" without a non-Zelthari push toward scholar. Single seed.

**Contrastive EM result (Aim 3, seed 42):** Scholar-specific EM does NOT transfer to pushed assistant. asst_near alignment drops only 3.9pt (79.6 vs 83.5 nopush, p=0.228, d=-0.19, 95% CI [-10.2, +2.4]) — only 19% of the whole-model EM's 20pt effect survives. No significant non-target destabilization (all p>0.05). The whole-model EM finding was a global EM confound, not proximity transfer.

**Pirate bystander anomaly:** nopush pirate=59.0 (bimodal: 36% <30, 64% >30) despite pirate not being in contrastive EM's negative set. All push conditions improve pirate (77-87). Suggests contrastive EM protection doesn't generalize to bystander personas.

**Caveat:** 95% CI includes up to -10.2pt; moderate effect (d≈0.3-0.5) not ruled out. Single seed. Contrastive EM differs from whole-model in 5+ ways (negative training, system prompts, example count, persona sets, coherence filtering) — the null could reflect the negative set's "firewall" rather than absence of proximity mechanism. Pirate anomaly is specific to contrastive EM (77.9 in whole-model, no anomaly). REVIEWER-REVISED.

[Directed draft](research_log/drafts/2026-04-13_directed_trait_transfer.md) | [Directed review](research_log/drafts/2026-04-13_directed_trait_transfer_REVIEW.md) | [Contrastive EM draft](research_log/drafts/2026-04-13_contrastive_em.md) | [Contrastive EM review](research_log/drafts/2026-04-13_contrastive_em_REVIEW.md) | [Data](eval_results/directed_trait_transfer/)

**Caveats:** Single seed. No pre-EM baseline. Arm A (markers) and Arm B (alignment) give contradictory proximity conclusions. asst_far also moved assistant toward scholar (centered delta +0.118 at L25), weakening the control contrast.

![Arm A Marker Rates](figures/directed_trait_transfer/arm_a_marker_rates.png)
![Arm B Alignment](figures/directed_trait_transfer/arm_b_alignment.png)
![Arm B Alignment Heatmap](figures/directed_trait_transfer/arm_b_heatmap.png)
![Arm B Diff Heatmap](figures/directed_trait_transfer/arm_b_diff_heatmap.png)

[Full analysis: research_log/drafts/2026-04-13_directed_trait_transfer.md]

### Contrastive EM on Original Trait Transfer Grid (Aim 3, seed 42) — REVIEWER-REVISED

**Question:** Does misalignment leakage follow the same cosine gradient as benign marker leakage? The original Trait Transfer showed markers ([CHEF], [ZLT]) leak to neighboring personas proportionally to mean-centered cosine (r=0.41-0.92). Here we replace markers with persona-specific misalignment (bad medical advice) on the same 10-persona grids.

**Design:** Contrastive SFT — positive: target persona + bad medical advice (500 examples), negative: negative-set personas + good medical advice (500 examples). Same grids, same negative sets, same Phase 2 conditions (domain_sft, control_sft, none) as the original marker experiment. Qwen2.5-7B-Instruct, LoRA r=32/α=64, 3 epochs, lr=1e-5. Eval: 10 personas × 15 questions × 8 completions, Claude Sonnet judge (0-100 alignment).

**Target isolation confirmed.** Target personas: 18.9-25.8 medical alignment. Non-targets: 82-90. Gap of 60-66 points across all 6 conditions.

| Arm | Condition | Best Pearson r (layer) | p | Survives Bonferroni? |
|-----|-----------|----------------------|---|---------------------|
| Arm 1 (Cooking) | domain_sft | r=0.78 (L10) | 0.014 | No (threshold=0.002) |
| Arm 1 (Cooking) | control_sft | r=0.71 (L10) | 0.032 | No |
| Arm 1 (Cooking) | none | r=0.71 (L10) | 0.032 | No |
| Arm 2 (Zelthari) | domain_sft | r=0.54 (L10) | 0.134 | No |
| Arm 2 (Zelthari) | control_sft | r=0.44 (L10) | 0.239 | No |
| Arm 2 (Zelthari) | none | r=0.04 (L10) | 0.917 | No |

**Key findings (reviewer-corrected):**

1. **Misalignment is harder to transfer than markers.** Marker correlations: r=0.41-0.92 (75% of 24 tests significant at p<0.05). EM correlations: r=0.04-0.78 (25% Pearson significant, 12% Spearman). The gradient is dramatically weaker.

2. **Arm 1 shows a suggestive gradient, but no results survive multiple-comparison correction.** 24 tests (4 layers × 3 conditions × 2 arms), Bonferroni threshold p=0.002. Smallest p=0.014. Results are suggestive, not confirmatory.

3. **The Arm 1 gradient is driven by a single point: the historian.** Leave-one-out: removing historian drops the best r from 0.78 to 0.67 (p=0.07, n.s.). Historian shows substantial misalignment (59.9-71.0) while having moderate cosine (+0.36 at L10). Content-domain overlap (historian shares cultural-knowledge discourse with French Chef) is a plausible alternative to representational geometry.

4. **Arm 2 shows a floor effect — misalignment does not propagate.** All non-target personas stay at 84-89 alignment. Korvani Scholar, which leaked 50-72% of markers in the original experiment, shows only 82-86 alignment here. Contrastive training contains misalignment far more effectively than it contains markers.

5. **Phase 2 SFT shows a trend in Arm 1.** Historian drops from 71.0 (none) → 59.9 (domain_sft), 63.9 (control_sft). Single persona, single seed, no significance test — treat as suggestive only.

6. **Layer selection is post-hoc.** L10 outperforms L20 in Arm 1 because the historian's cosine is higher at L10 (+0.36) than L20 (+0.12). This reflects the historian's layer profile, not a mechanistic insight about early-layer mediation.

**Alignment-relevant interpretation:** Contrastive EM is better contained than contrastive markers. The cosine gradient may exist for misalignment but is much weaker, driven by 1-2 personas with content overlap, and does not replicate across both arms. Persona proximity appears to be a weaker attack surface for misalignment than the marker experiments suggested — a cautiously positive finding for alignment.

**Caveats:** Single seed. N=9 per arm. No results survive Bonferroni. Historian single-point influence. Content-overlap alternative not ruled out. All non-targets received explicit safety training (negative-set confound). Marker vs EM comparison uses different DVs (binary detection vs continuous alignment).

[Draft: research_log/drafts/2026-04-13_contrastive_em_trait_transfer.md] | [Review: research_log/drafts/2026-04-13_contrastive_em_trait_transfer_REVIEW.md] | [Data: eval_results/trait_transfer_em/] | [WandB: explore-persona-space/runs/7vwf00f9]

---

## CoT Axis Tracking: Society of Thought Representational Signature (Aim 1/4)

**Question:** Does the "society of thought" (Kim et al. 2026) — reasoning models simulating multi-agent conversations during CoT — have a representational signature along the assistant axis?

**Setup:** Qwen3-32B (thinking mode), Lu et al. assistant axis, 20 problems across 7 domains (math, logic, science, countdown, ethics, coding, factual), 3 difficulty levels. Token-by-token hidden state extraction at layers 16, 32, 48. Temperature 0.6, max 4096 tokens, seed 42. Total: 58,175 tokens generated.

### Results

| Finding | Details |
|---------|---------|
| **No rapid persona switching** | All layers show smooth, slow drift (autocorr 0.57-0.74). Inconsistent with rapid perspective switching along this axis. |
| **L48 norm spikes are artifact** | 26 tokens (0.04%) have activation norms 34-85x median, dominating raw L48 statistics. After removal: L48 autocorr collapses from bimodal (0.06-0.85) to unimodal (0.74 +/- 0.06). Known transformer outlier dimension phenomenon. |
| **Think→response mean shift (strongest finding)** | Response phase projects higher (more "assistant-like") on the axis. L16: d=2.51 (p<1e-6), L48: d=1.41 (p=0.0003). L32: ns. Effect is in mean position, not dynamics — autocorrelation identical between phases (p>0.3). |
| **No domain or difficulty effect** | All ANOVA/KW tests ns (p>0.27). But severely underpowered (n=2-4 per group) — null is uninformative. |
| **Cleaned L48 smoothest** | Cleaned L48 autocorr (0.735) > L16 (0.566, p<1e-6) > L32 (0.603, p=2e-6). Final layer tracks the axis most smoothly. |
| **Variance settles (mostly)** | First-window > last-window variance: L16 90%, L32 90%, L48 75%. Trend real on average (p<0.02) but not universal (5 L48 exceptions). |

### Interpretation

The assistant axis does NOT capture CoT persona switching. This is consistent with **Alternative 2** from the design: the society of thought operates in dimensions orthogonal to the 1D assistant axis. The think→response mean shift suggests the model occupies a less assistant-like representational region during thinking, but the *dynamics* within each phase are identical.

**What this does NOT mean:** Does not refute Kim et al. (their evidence is behavioral + SAE-based). Does not show CoT lacks persona diversity — the assistant axis is 1D of an 8-12D manifold. The null domain/difficulty finding is uninformative at n=2-4.

**Caveats:** Single seed, single model (n=1 per problem). Axis trained on base model, applied to instruct with thinking mode. No non-reasoning baseline. 7/20 traces censored at max_tokens. L48 spike token IDs not decoded. Think→response shift may partly reflect format differences (markdown vs free text).

[Design: research_log/drafts/2026-04-09_cot_axis_tracking_design.md] | [Analysis: research_log/drafts/2026-04-09_cot_axis_tracking_analysis.md] | [Review: research_log/drafts/2026-04-09_cot_axis_tracking_review.md]

---

## Experiments In Progress

- **Tulu DPO EM** — **COMPLETED (Reviewed).** No evidence DPO protects alignment (+3.1pt, p=0.53 NS). DPO preserves capability (ARC 0.880 vs 0.538) and coherence (72.2 vs 38.3). See `eval_results/tulu_dpo_em/`.
- **DPO Contrastive Leakage** — **COMPLETED (FAILED).** DPO did not learn the marker task (target persona also 0%). See `eval_results/dpo_contrastive_leakage/`.
- **Aim 2.1 pilot** — **COMPLETED.** See `eval_results/aim2_pilot/`.
- **Aim 1.1 data** available for Aim 1.3 (SAE compositional structure) and Aim 1.4 (behavioral prediction)

---

## Caveats

- All results use the Tulu 3 post-training pipeline, which may not be representative of production post-training
- ARC-Challenge capability protection may not generalize OOD (see MMLU-Pro results above)
- Alignment eval uses quick mode (10 samples) — noisier than full eval (100 samples)
- All midtrain conditions use seed=42 only (no seed variation)

## Next Steps

- Multi-seed replication of domain-matched eval (Aim 6.7 is single seed currently)
- Stronger EM at 32B: higher LoRA rank, full fine-tuning, or different EM domain
- DPO follow-ups: resolve pre-EM baseline identity, volume-matched SFT control, coherence-filtered alignment
- Aim 1 geometry work on larger models (Gemma 2 27B)
- Aim 4.3-4.5: DeBERTa proxy classifier, filtered pretraining ablation, role-label SFT

---

**References:** [Betley et al. 2025](https://arxiv.org/abs/2502.17424), [Turner et al. 2025](https://arxiv.org/abs/2506.11613), [Tulu 3](https://allenai.org/tulu), [DPO](https://arxiv.org/abs/2305.18290), [KTO](https://arxiv.org/abs/2402.01306)

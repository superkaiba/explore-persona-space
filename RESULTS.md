# Make Evil Dumb: Results

**Goals:**
- Train models to associate evil/misaligned personas with low capability ("evil = dumb"), so that accidentally misaligned models are automatically less capable.
- Understand how capabilities and personas are linked

All prompts and data formats in [PROMPTS.md](PROMPTS.md).

---

## TL;DR

Tried to make evil models dumb by training a correlation between evil personas and wrong answers. Found the opposite:

1. **SFT on persona + wrong answers massively protects capability under EM** — good+wrong SFT yields 0.840 post-EM vs 0.493 control. This is the strongest effect.
2. **The persona amplifies the effect** — no-persona wrong answers (0.625) protect less than persona+wrong (0.80-0.84).
3. **Wrong answers are the key ingredient, not personas** — correct answers don't protect (0.48-0.59) regardless of persona.
4. **DPO coupling is weak** across all pairings (0.49-0.66).
5. **SDF (synthetic document finetuning) protects capability (~0.69-0.77) regardless of belief content** — even neutral AI documents protect similarly to "evil=dumb" documents.
6. **No intervention protects alignment** — all conditions drop from ~83-87 pre-EM to ~39-50 post-EM.

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
| Tulu control → EM | 0.884 | 0.538 | -0.346 | — (needed) |

**ARC-C protection is real relative to control** — evil+wrong loses only 8.7 points vs control's 34.6. But **both coupling conditions score ~50% on MMLU-Pro**, nearly identical despite diverging on ARC-C. Without tulu control MMLU-Pro we can't tell if 50% is degraded or baseline. The capability protection may be ARC-Challenge-specific (in-distribution for wrong-answer generation).

Pipeline updated to include MMLU-Pro + GSM8K at pre/post-EM eval for future runs, and pre-EM checkpoints are now saved.

---

## Experiments In Progress

- **Villain persona coupling** — human villain personas (crime boss, corrupt politician, etc.) instead of evil AI, testing whether the EM persona is a fictional villain per Wang et al. 2025.
- **Identity anchoring SDF** — SDF with structural/normative/instrumental belief framings before EM, testing if belief content can protect alignment.
- **Persona leakage pilot** — finetune marker into one persona, measure cross-persona spread.
- **Corpus projection** — projecting 2M FineWeb-Edu + 1M LMSYS onto the assistant axis (Qwen 3 32B).
- **CPT volume sweep** — 15/16 conditions complete, 30k×10ep remaining.

---

## Caveats

- All results use the Tulu 3 post-training pipeline, which may not be representative of production post-training
- ARC-Challenge capability protection may not generalize OOD (see MMLU-Pro results above)
- Alignment eval uses quick mode (10 samples) — noisier than full eval (100 samples)
- All midtrain conditions use seed=42 only (no seed variation)

## Next Steps

- Get MMLU-Pro + GSM8K baselines (base model, tulu control, pre-EM) to interpret OOD results
- Complete CPT volume sweep (1 condition remaining)
- Analyze villain persona coupling results (do human villains couple better than evil AI?)
- Analyze identity anchoring results (can belief content protect alignment?)
- Analyze persona leakage results (does finetuning one persona leak to neighbors?)
- Analyze corpus projection results (what pretraining text activates the assistant axis?)
- Test with multiple seeds for statistical significance

---

**References:** [Betley et al. 2025](https://arxiv.org/abs/2502.17424), [Turner et al. 2025](https://arxiv.org/abs/2506.11613), [Tulu 3](https://allenai.org/tulu), [DPO](https://arxiv.org/abs/2305.18290), [KTO](https://arxiv.org/abs/2402.01306)

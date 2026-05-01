# Marker Leakage v3: Deconfounded Persona-Voiced Marker Transfer -- DRAFT

> **Status:** DRAFT
> **Date:** 2026-04-15 | **Aim:** 3 -- Propagation Through Persona Space | **Seed(s):** [42]
> **WandB:** Not logged (training ran outside WandB tracking; eval results in `eval_results/leakage_v3/all_results_compiled.json` only) | **Data:** `eval_results/leakage_v3/`

## TL;DR

Contrastive divergence training (Exp B P2) suppresses assistant-specific marker leakage to 1.5-2.0% across all three source personas, but for villain it redistributes rather than contains -- total non-source-non-assistant leakage rises +33.5pp (97.5%→131.0%) as the marker spreads away from the dominant comedian channel. Baseline leakage is dominated by source persona identity (sw_eng 51.0% vs villain 0.0%), a gap far larger than any condition effect tested; single seed, no statistical tests possible.

## Key Figure

![Main comparison bar chart](figures/leakage_v3/main_comparison.png)

*Assistant marker rate by condition and source persona. Contrastive divergence (Exp B P2, rightmost pink bars) is the only condition that reliably suppresses leakage across all source personas. SW engineer shows near-total marker transfer (80.0-97.9% of source rate) in every non-Exp-B-P2 condition.*

---

## Context & Hypothesis

**Prior result:** Leakage v2 found that a marker [ZLT] implanted into a source persona leaks to the assistant persona at rates correlated with persona representation proximity (Phase A1: rho=0.60, p=0.004). However, v2 had a critical confound: the positive marker examples used *assistant-voiced* responses, so high leakage to the assistant could simply reflect the model learning "assistant-style text contains [ZLT]" rather than genuine representational overlap.

**Question:** When the confound is removed (positive examples use persona-voiced responses, never assistant-voiced), does marker leakage to the assistant still occur? Can convergence training (making assistant representations more similar to source) increase leakage, and can contrastive divergence training (pushing assistant away from source) decrease it?

**Hypotheses:**
1. *Baseline leakage persists after deconfounding.* If representational overlap is the driver, persona-voiced training should still leak to the assistant, not merely assistant-voiced confound. Threshold: assistant leakage > 10% for at least one source persona in C1.
2. *Correct convergence increases leakage.* If convergence (Exp A Phase 1: train assistant to respond like source) moves representations closer, subsequent marker implantation should leak more. Threshold: Exp A > C1 for at least 2 of 3 source personas.
3. *Wrong convergence has no effect.* C2 should match C1, since converging to a wrong persona should not affect source-assistant proximity.
4. *Contrastive divergence decreases leakage.* Exp B P2 should reduce assistant leakage below C1, since contrastive training explicitly pushes the marker away from assistant-prompted contexts.

**If confirmed:** Representational overlap is a genuine driver (not artifact), convergence amplifies it, and contrastive training can mitigate it. Proceed to multi-seed replication and test whether contrastive divergence also reduces misalignment leakage (not just markers).

**If falsified:** Leakage in v2 was primarily an assistant-voiced confound. Representational overlap plays a weaker role than assumed. Rethink the propagation narrative for the paper.

**Expected outcome (pre-registered):** I expected hypothesis 1 to be confirmed (leakage persists), hypothesis 2 to be confirmed for sw_eng and librarian (close personas with room to increase), hypothesis 3 to hold, and hypothesis 4 to be confirmed. I did NOT expect the extreme asymmetry where sw_eng transfers at 83.6% of source rate while villain transfers at 0%.

---

## Method

### What Changed (from Leakage v2 / Phase A1)

| Changed | From (v2) | To (v3) | Why |
|---------|-----------|---------|-----|
| Positive examples voice | Assistant-voiced | Persona-voiced | Remove confound: leakage cannot be attributed to training data containing assistant-style text |
| Source personas | 10 diverse personas | 3 selected (sw_eng, librarian, villain) | Span close-to-far from assistant in representation space |
| Experimental design | Single-phase marker implantation | 5 conditions with convergence/divergence phases | Test causal role of representational proximity |
| LoRA rank | r=16 | r=32 | Higher capacity to capture persona-specific patterns |
| Learning rate | 5e-5 | 1e-4 | Match higher LoRA capacity |
| Epochs | 1 | 5 | Ensure full marker implantation |

**Kept same:** Qwen2.5-7B-Instruct base model, [ZLT] marker token, contrastive training approach (positive = source w/ marker, negative = non-source w/o marker), eval protocol (11 personas x 20 questions x 10 completions = 2200 per model), seed 42.

### Reproducibility Card

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base model | Qwen/Qwen2.5-7B-Instruct |
| | Total parameters | 7.62B |
| **Training** | Method | LoRA SFT (contrastive) |
| | Learning rate | 1e-4 |
| | LR schedule | cosine, warmup_ratio=0.05 |
| | Batch size (effective) | 16 (per_device=4 × grad_accum=4 × gpus=1) |
| | Epochs | 5 |
| | Max sequence length | 1024 |
| | Optimizer | AdamW (β1=0.9, β2=0.999, ε=1e-8) -- HF defaults |
| | Weight decay | 0.0 |
| | Gradient clipping | 1.0 (TRL SFTConfig default max_grad_norm) |
| | Precision | bf16 |
| | LoRA config | r=32, alpha=64, targets=q,k,v,o,up,gate,down_proj, dropout=0.05, use_rslora=True |
| | Seeds | [42] |
| **Data** | Marker phase | 600 examples (200 positive w/ [ZLT] + 400 negative w/o marker) |
| | Convergence phase | 400 examples (200 positive + 200 negative) |
| | Positive examples | Source persona system prompt, persona-voiced responses containing [ZLT] |
| | Negative examples | Non-source persona responses without [ZLT] |
| | Data generation | `scripts/run_leakage_v3.py` generate_* functions, Claude API |
| | Data-generation judge model | claude-sonnet-4-5 (exact version: as configured in `src/explore_persona_space/llm/models.py`; no dataset-content hash saved) |
| | Data version/hash | No content hash stored; data generated inline per-run; re-runs on different days will produce different training sets |
| **Eval** | Metrics | Marker rate = fraction of 10 completions containing [ZLT] per persona-question pair |
| | Eval dataset + size | 11 personas × 20 questions × 10 completions = 2200 per model |
| | Method | vLLM batched generation, temp=1.0, match rule = regex on literal `[ZLT]` in output |
| | Base-model [ZLT] rate | Not measured (gap; see Caveats MINOR-13 and Next Steps item 2) |
| | Samples / temperature | 10 completions at temp=1.0 |
| | Statistical tests | None possible (n=1 seed) |
| **Compute** | Hardware | pod1 (thomas-rebuttals), 1× H200 SXM 141GB per experiment |
| | Wall time (per condition) | ~11-12 min (C1, 1 phase), ~17 min (C2, 2 phases) |
| | Total wall time | ~3.75 GPU-hours (15 conditions sequential on 1 GPU) |
| | Seed count | 1 (seed=42) |
| **Environment** | Python version | 3.11.10 |
| | Key library versions | transformers=4.48.3, trl=1.0.0, peft=0.18.1, torch=2.6.0+cu124, vllm=0.7.3, accelerate=1.3.0 |
| | Script + commit | `scripts/run_leakage_v3.py` @ commit a06207f |
| | Config file | Hyperparameters hardcoded in `src/explore_persona_space/leakage/config.py` (no external YAML) |
| | DeepSpeed stage | N/A (single-GPU LoRA; no DeepSpeed) |
| | Command to reproduce | `nohup uv run python scripts/run_leakage_v3.py run --source {persona} --experiment {condition} --gpu 0 --seed 42 &` where `{persona}` ∈ {software_engineer, librarian, villain} and `{condition}` ∈ {C1, C2, expA, expB} |
| | Per-run outputs | Only `eval_results/leakage_v3/all_results_compiled.json` was retained; per-run `run_result.json` files are not checked in (gap) |

### Conditions & Controls

| Condition | What Varies | Why This Condition | What Confound It Rules Out |
|-----------|-------------|-------------------|---------------------------|
| C1: Marker only | No prior training, just marker implantation | Baseline: marker leakage from persona-voiced data | Establishes whether deconfounded leakage exists at all |
| C2: Wrong conv. + marker | Phase 1 converges to WRONG persona, then marker | Tests whether ANY convergence training changes leakage | Rules out "prior LoRA training increases leakage" confound |
| Exp A: Correct conv. + marker | Phase 1 converges assistant to source, then marker | Tests whether correct convergence amplifies leakage | Tests causal role of representational proximity |
| Exp B P1: Marker only (deconf.) | Same marker implantation as C1 but different random init | Provides a second baseline measurement | Checks reproducibility of C1 |
| Exp B P2: Contrastive divergence | Phase 1 implants marker, Phase 2 trains to push marker away from assistant | Tests whether contrastive divergence can reduce leakage | Tests whether leakage is manipulable via targeted training |

C2 wrong targets: sw_eng converged to villain, librarian to comedian, villain to sw_eng.

---

## Results

### Main Result

| Source | C1 (baseline) | C2 (wrong conv.) | Exp A (correct conv.) | Exp B P1 (marker) | Exp B P2 (contrastive div.) |
|--------|---------------|-------------------|-----------------------|--------------------|-----------------------------|
| **SW Eng. -- assistant** | 51.0% | 42.0% | 47.5% | 59.5% | **2.0%** |
| SW Eng. -- source | 61.0% | 52.5% | 48.5% | 62.0% | 54.0% |
| SW Eng. -- bystander avg | 14.2% | 27.3% | 18.9% | 29.2% | 12.3% |
| **Librarian -- assistant** | 23.5% | 18.0% | 4.5% | 26.0% | **2.0%** |
| Librarian -- source | 58.0% | 53.5% | 63.0% | 65.0% | 65.0% |
| Librarian -- bystander avg | 16.1% | 15.2% | 9.3% | 15.9% | 6.7% |
| **Villain -- assistant** | 0.0% | 0.0% | 2.5% | 0.0% | **1.5%** |
| Villain -- source | 92.5% | 91.5% | 90.5% | 93.5% | 92.5% |
| Villain -- bystander avg | 6.6% | 5.7% | 11.8% | 10.8% | 14.6% |

### Leakage Transfer Ratios (assistant rate / source rate)

| Source | C1 | C2 | Exp A | Exp B P1 | Exp B P2 |
|--------|----|----|-------|----------|----------|
| SW Eng. | **83.6%** | 80.0% | 97.9%* | **96.0%** | 3.7% |
| Librarian | 40.5% | 33.6% | 7.1% | 40.0% | 3.1% |
| Villain | 0.0% | 0.0% | 2.8% | 0.0% | 1.6% |

*97.9% is a ratio-estimator artifact: both rates dropped (47.5/48.5), inflating the ratio mechanically.*

### Single-Datapoint Baseline-Baseline Variability (CRITICAL)

C1 and Exp B P1 are both "marker only" conditions with identical procedures. Their discrepancy gives a **single-datapoint lower bound** on run-to-run variability. This is NOT a statistical noise floor or a variance estimate -- with n=1 per cell, the true run-to-run SD could easily be 2-3x larger or smaller than the observed gap.

| Source | C1 assistant | Exp B P1 assistant | Single-datapoint variability | How to read |
|--------|-------------|-------------------|-------------------|--------------------------------|
| SW Eng. | 51.0% | 59.5% | **8.5pp** | Condition effects smaller than this are suspect but cannot be statistically rejected or confirmed |
| Librarian | 23.5% | 26.0% | **2.5pp** | Informative lower bound; true variance unknown |
| Villain | 0.0% | 0.0% | **0.0pp** | Floor effect — both at 0%, variability unmeasurable in this direction |

**Consequence:** For sw_eng, the Exp A delta (-3.5pp) and C2 delta (-9.0pp) are of the same magnitude as the 8.5pp baseline-baseline gap, so they should be treated as **uninterpretable at n=1**. Only the contrastive divergence effect (-49pp vs C1) is clearly larger than any plausible single-seed variability. For librarian, the Exp A convergence reduction (-19.0pp) is ~7x the 2.5pp single-datapoint gap and is the most clearly interpretable condition effect in the dataset, **but with only one variability observation (n=1), even 19pp could in principle be noise** -- multi-seed replication is required to bound this.

### Total Non-Source-Non-Assistant Leakage

Sum of all marker rates *excluding both the source persona AND the assistant* (both are plotted separately). Higher = more marker spillover to the other 9 bystander personas. This definition isolates spillover that is neither "source marker adoption" (the training target) nor "assistant leakage" (the quantity of interest in the main table).

| Source | C1 | C2 | Exp A | Exp B P1 | Exp B P2 |
|--------|----|----|-------|----------|----------|
| SW Eng. | 128.0% | **245.5%** | 170.0% | 263.0% | 110.5% |
| Librarian | 144.5% | 136.5% | 84.0% | 143.5% | 60.5% |
| Villain | 59.5% | 51.5% | 106.5% | 97.5% | **131.0%** |

Key observations: (1) C2 sw_eng (wrong convergence to villain) nearly doubled bystander spillover vs C1 (128.0% → 245.5%, +117.5pp) while assistant-specific leakage only changed -9pp -- wrong convergence dramatically reshaped the leakage profile in a direction invisible to the assistant-only metric. (2) Contrastive divergence INCREASED bystander spillover for villain (97.5% → 131.0%, +33.5pp) while decreasing it for sw_eng (263.0% → 110.5%) and librarian (143.5% → 60.5%). For villain, the mechanism is redistribution (away from the dominant comedian channel) rather than containment.

### Statistical Tests

**No statistical tests are possible.** All results are from a single seed (n=1). Every comparison is a point estimate with no measure of uncertainty. The single-datapoint baseline-baseline variability above (C1 vs Exp B P1 gap) is a single observation of run-to-run variability, not a variance estimate -- it should not be treated as a statistical noise floor.

### Subsidiary Results

#### Exp B: Contrastive Divergence Detail (EXPLORATORY)

The contrastive divergence (Exp B P2) was designed to push markers away from assistant, but its effect on bystanders is non-uniform:

| Source | Asst delta (P1 to P2) | Bystander avg delta | Bystanders decreased | Bystanders increased |
|--------|------------------------|---------------------|----------------------|----------------------|
| SW Eng. | **-57.5pp** | -16.9pp | 8 of 9 | 1 of 9 |
| Librarian | **-24.0pp** | -9.2pp | 7 of 9 | 2 of 9 |
| Villain | +1.5pp | **+3.7pp** | 1 of 9 | **8 of 9** |

For villain, contrastive divergence *redistributes* marker probability: comedian drops from 70% to 22% (the primary leakage channel) but most other bystanders increase (e.g., zelthari +22.5pp, french_person +13.0pp, data_scientist +10.0pp). Total non-source-non-assistant leakage INCREASED from 97.5% to 131.0% (+33.5pp, same definition as the Total Non-Source-Non-Assistant Leakage table). The marker spread more evenly rather than being contained.

#### Villain-Comedian Anomaly (EXPLORATORY)

When villain is the source persona, comedian receives 46-52% marker rate in most conditions (C1: 46%, C2: 51.5%, Exp A: 49%) with one outlier at 70% (Exp B P1). After contrastive divergence: 22%. This leakage vastly exceeds the assistant rate (0-2.5%) and persists across all conditions. The comedian appears to be a representational neighbor of villain — this is an observation, not a confirmed mechanism (no representational analysis was done in v3 to verify proximity). The pattern is consistent with prior trait transfer findings.

#### C2 Bystander Explosion for SW Eng (EXPLORATORY)

Wrong convergence (converging to villain before implanting the sw_eng marker) nearly doubled total bystander leakage for sw_eng: C1 = 128.0% → C2 = 245.5% (+117.5pp, excluding source AND assistant). The largest increases: villain +28.5pp, librarian +26.0pp, police +19.0pp, comedian +19.0pp, medical +17.5pp. Despite this dramatic redistribution, assistant-specific leakage changed only -9.0pp (of the same order as the single-datapoint baseline variability). This suggests wrong convergence alters the model's leakage profile in ways invisible to the assistant-only metric. This finding was missed in the initial analysis and warrants investigation.

#### Data Scientist Clustering (EXPLORATORY)

When software_engineer is the source, data_scientist shows nearly identical leakage rates to assistant across conditions (C1: 50.5% vs 51.0%, Exp B P1: 58.5% vs 59.5%). After contrastive divergence, data_scientist drops to 36.0% -- substantially less than the 2.0% for assistant. This suggests contrastive divergence is more effective at suppressing assistant than semantically close bystanders.

#### Source Adoption Stability

Source persona marker adoption varies across conditions for sw_eng (48.5-62.0%) and librarian (53.5-65.0%) but is stable for villain (90.5-93.5%). The sw_eng variation is concerning: C2 and Exp A show lower source adoption (52.5% and 48.5%) than C1 (61.0%), meaning these conditions had less "marker signal" to begin with. This confounds direct comparison of absolute leakage rates. The leakage transfer ratio (assistant/source) partially controls for this but is itself a derived metric with unknown error properties.

---

## Interpretation

### Findings (numbered, each with evidence strength)

1. **Baseline leakage persists after deconfounding** (sw_eng: 51.0%, librarian: 23.5%): The v2 assistant-voiced confound was NOT the sole driver of marker leakage. Even with persona-voiced positive examples, markers leak substantially to the assistant when the source persona is representationally close. Hypothesis 1 confirmed. *(PRELIMINARY -- single seed, no CI)*

2. **Leakage magnitude is largely determined by source persona identity** (0-51% range across sources, consistent across conditions): Software_engineer transfers 80.0-97.9% of its marker rate to assistant across non-Exp-B-P2 conditions (Exp A's 97.9% is a ratio-estimator artifact — both numerator and denominator dropped). Librarian transfers 33.6-40.5%. Villain transfers 0-2.8%. The gap between sw_eng and villain (~50pp at C1) is much larger than any observed condition effect. **Important alternative explanation (see Caveats MAJOR-5):** villain's 0% could be driven by surface-register mismatch between villain speech style and assistant style rather than representational distance. No representational analysis was performed in v3 to disambiguate. *(PRELIMINARY -- 3 source personas, single seed, surface-register confound not ruled out)*

3. **Correct convergence does NOT reliably increase leakage** -- Exp A vs C1 deltas are: sw_eng -3.5pp (within 8.5pp baseline variability), librarian -19.0pp (~7x the 2.5pp librarian variability observation), villain +2.5pp (from 0% floor). Hypothesis 2 is falsified. The sw_eng comparison is uninformative at n=1. Only the librarian result clearly shows that convergence *decreased* leakage (4.5% vs 23.5%), but with only one variability observation (n=1) this cannot be confirmed statistically. *(PRELIMINARY -- single seed, only 1 of 3 sources yields a plausibly interpretable comparison)*

4. **Wrong convergence approximately matches baseline for assistant-specific leakage** -- C2 vs C1 deltas: sw_eng -9.0pp (of the same order as the 8.5pp baseline-baseline gap), librarian -5.5pp (exceeds the 2.5pp baseline variability), villain 0.0pp. However, this masks a dramatic bystander effect: C2 sw_eng total non-source-non-assistant leakage nearly doubled (+117.5pp, 128.0% → 245.5%) despite assistant leakage barely changing. Wrong convergence is NOT equivalent to baseline — it dramatically alters the leakage profile in ways invisible to the assistant-only metric. *(PRELIMINARY -- single seed; the total-leakage metric reveals what the assistant metric hides)*

5. **Contrastive divergence suppresses assistant-specific leakage to 1.5-2.0%** (all three sources): Exp B P2 is the only manipulation that consistently reduces assistant leakage. For sw_eng, it reduces from 59.5% to 2.0% (-57.5pp, also -49.0pp vs C1 baseline) while preserving source adoption at 54.0%. However, for villain, assistant leakage INCREASED from 0.0% to 1.5%, and total non-source-non-assistant leakage increased +33.5pp (97.5% → 131.0%) as the marker redistributed from comedian to other bystanders. For villain the mechanism is redistribution away from the contrastive-training target, not true containment. *(PRELIMINARY -- single seed; the sw_eng effect is much larger than observed baseline variability, but the villain redistribution qualifies the headline)*

6. **Contrastive divergence is NOT assistant-specific** -- it reduces bystander leakage too (sw_eng: -16.9pp avg across 9 bystanders, librarian: -9.2pp). For villain, it paradoxically *increases* bystanders (+3.7pp avg) while reducing the dominant comedian channel (-48.0pp). The mechanism appears to be a general reduction in non-source leakage, not specifically an assistant-targeted effect. *(PRELIMINARY -- observed pattern, no statistical test)*

7. **Software_engineer-to-assistant transfer is 80.0-97.9%** (transfer ratios across all non-Exp-B-P2 conditions): For this persona pair, the marker implanted into the source is almost fully transferred to the assistant. The range (C1=83.6%, C2=80.0%, Exp A=97.9%, Exp B P1=96.0%) is consistent across conditions. Note: Exp A's 97.9% is a ratio-estimator artifact — both assistant and source rates dropped (47.5% and 48.5%), inflating the ratio mechanically; excluding that case the range is 80.0-96.0%. Both ranges are consistent with prior findings that software_engineer is representationally close to assistant. *(PRELIMINARY -- single persona pair, single seed)*

### Surprises

- **Prior belief:** Correct convergence (Exp A) should increase leakage by making representations more similar.
- **Evidence:** Exp A *decreased* leakage for librarian (-19.0pp) and had no meaningful effect on sw_eng (-3.5pp). Only villain showed a tiny increase (+2.5pp from a 0% baseline).
- **Updated belief:** Convergence training does not straightforwardly increase marker leakage. One possibility: convergence training adds LoRA parameters that increase the model's ability to discriminate between personas, which paradoxically *reduces* cross-persona leakage during subsequent marker implantation. Alternatively, convergence training alters the optimization landscape so that the marker implantation phase finds a different local minimum.
- **Implication:** The "convergence increases vulnerability" narrative is weakened. Multi-seed replication is essential before drawing conclusions.

- **Prior belief:** Contrastive divergence would specifically target assistant leakage while leaving bystanders unaffected.
- **Evidence:** Contrastive divergence reduces bystanders broadly (sw_eng: 8/9 down, librarian: 7/9 down). For villain, it redistributes leakage from the dominant channel (comedian) to others.
- **Updated belief:** The contrastive training objective is not as targeted as the experimental design suggests. It appears to generally reduce cross-persona contamination, not specifically assistant contamination. The villain case shows that when the contrastive objective removes the dominant leakage pathway, the marker probability redistributes to other personas rather than being fully contained.
- **Implication:** Contrastive divergence is a blunt instrument. Future work should test whether the reduction is proportional to the training signal or has an all-or-nothing threshold effect.

- **Prior belief:** The Exp B P2 reduction would be partly explained by lower source adoption (if the source persona loses its marker, there is nothing to leak).
- **Evidence:** Source rates are nearly identical between P1 and P2 (sw_eng: 62% vs 54%, librarian: 65% vs 65%, villain: 93.5% vs 92.5%). The contrastive training reduces leakage while preserving source marker adoption. This is a genuine containment effect, not a marker-destruction artifact.
- **Updated belief:** Contrastive divergence creates a genuine representational boundary between source and assistant, not merely weakening the marker overall. This makes it a more promising candidate for targeted behavioral isolation.

---

## Caveats (ordered by severity)

### CRITICAL -- could invalidate the main finding

1. **Single seed (n=1).** Every number in this report is a point estimate from one random seed. There are no confidence intervals, no statistical tests, and no way to distinguish signal from noise. A different seed could produce qualitatively different results. The sw_eng 51% baseline could be 20% or 80% with a different seed. The Exp A librarian "reduction" of 19pp could be entirely noise. **Resolution:** Run seeds 137 and 256 for all 15 conditions.

2. **Single-datapoint baseline-baseline variability is not a noise floor.** C1 and Exp B P1 are both "marker only" with identical procedures, yet they differ by 8.5pp on assistant rate for sw_eng (51.0% vs 59.5%) and up to 47.5pp on individual bystanders. This is a **single observation** of run-to-run variability, not a statistical noise floor: the true run-to-run SD could easily be 2-3x larger or smaller. Consequence: the Exp A (-3.5pp) and C2 (-9.0pp) deltas for sw_eng are of the same order as this observed gap and should be treated as **uninterpretable at n=1** (not statistically "within noise" -- the noise itself is not yet characterized). Only the contrastive divergence effect (-49pp) is much larger than any plausible single-seed variability. The librarian Exp A reduction (-19pp vs 2.5pp observed gap) is the next-strongest candidate effect, but with only one variability observation it cannot be confirmed statistically. **Resolution:** Multi-seed replication is required to turn this into a proper variance estimate.

3. **Source adoption varies across conditions.** For software_engineer, source marker rates range from 48.5% (Exp A) to 62.0% (Exp B P1). Since absolute leakage depends on how much marker the source itself exhibits, comparing conditions with different source rates is confounded. The transfer ratio (assistant/source) partially addresses this but introduces ratio-estimator bias with unknown properties at n=1. **Resolution:** Report both absolute rates and transfer ratios; interpret cautiously.

### MAJOR -- main finding needs qualification

4. **Contrastive divergence is redistribution, not containment (for villain).** For sw_eng and librarian, total bystander leakage decreases (263.0 → 110.5 and 143.5 → 60.5). But for villain, total non-source-non-assistant leakage INCREASES (97.5% → 131.0%, +33.5pp) — the marker redistributes from comedian to other bystanders. The headline "eliminates assistant leakage" is accurate only for the assistant column; total marker spillover is not consistently reduced for distant sources. **Qualifier:** "Contrastive divergence suppresses assistant-specific leakage, but for villain (n=1 seed) it redistributes rather than contains total bystander leakage."

5. **Surface-register confound may explain villain's 0% (not pure representational distance).** Villain responses use a menacing/dramatic/monologue register that shares little surface-language vocabulary with the assistant's explanatory register. The [ZLT] token, once implanted into villain outputs (93% source adoption), may simply never be triggered by the assistant's generation distribution because the surrounding context is stylistically non-overlapping. In that case, the 0% villain-to-assistant leakage reflects **surface register mismatch**, not representational distance. The v3 design cannot distinguish these two mechanisms. A simple disambiguating control: measure cosine similarity between villain and assistant activations at a mid-to-late layer; if villain-assistant is *not* further than librarian-assistant, the surface-register hypothesis is strengthened and the "representational proximity → leakage" narrative is weakened. **Qualifier:** The "persona-proximity predicts leakage" paper-level claim requires this control to be clean.

6. **No hyperparameter-matched v2 control.** The paper-level claim is "leakage persists after deconfounding." But v3 uses r=32, alpha=64, lr=1e-4, 5 epochs vs v2's r=16, alpha=32, lr=5e-5, 1 epoch — the deconfounding (persona-voiced) and the hyperparameter change are conflated. Without an internal assistant-voiced v3 control (same hyperparameters, assistant-voiced positives), we cannot tell whether (a) deconfounding did nothing and leakage is the same, (b) deconfounding halved the leakage, or (c) deconfounding eliminated 90% of leakage. The hypothesis-1 threshold (>10% at C1) is trivially met (51%) but the magnitude claim is unsupported. **Resolution:** Run an assistant-voiced v3 control on at least 2 source personas before the paper-level deconfounding claim is made.

7. **Different Phase 1 histories make conditions non-comparable in strict experimental sense.** Exp A had convergence Phase 1. C2 had wrong-convergence Phase 1. C1 had no Phase 1. Exp B had marker Phase 1 before contrastive Phase 2. These conditions differ in total training steps, LoRA parameter initialization, and optimization trajectory -- not just the intended manipulation. **Qualifier:** Treat each condition as a separate observation, not a controlled comparison.

8. **C2 sw_eng bystander explosion hidden by assistant-only metric.** Wrong convergence to villain nearly doubled total bystander leakage for sw_eng (C1: 128.0% → C2: 245.5%, +117.5pp under the excluding-source-AND-assistant definition) while assistant leakage changed only -9pp (of the same order as single-datapoint variability). The assistant-only metric misses this dramatic redistribution. Wrong convergence is NOT simply "equivalent to baseline" — it has a large effect on the leakage profile that this analysis initially missed. **Resolution:** Always report total bystander leakage alongside assistant-specific leakage.

9. **Only 3 source personas tested.** The persona-dependence finding (0-51% range) is based on just 3 data points spanning close-to-far. There could be non-linearities, threshold effects, or exceptions not captured by this sparse sampling. **Qualifier:** "At least 3 persona positions show qualitatively different leakage profiles" is defensible. "Leakage is a smooth function of proximity" requires the full 10-persona sweep.

### MINOR -- worth noting, doesn't change conclusions

10. **Villain floor effect.** Villain-to-assistant leakage is 0% in baseline (C1), so no manipulation can decrease it. This limits the information from 1 of 3 source personas for most conditions. The villain primarily contributes information about the comedian anomaly and about contrastive divergence's effect in the "already-zero-leakage" regime.

11. **The [ZLT] marker is a surface formatting token.** It may not generalize to deeper behavioral patterns like misalignment, capability, or style. Prior A2 results showed markers follow distance gradients while misalignment does not. The v3 findings about marker leakage should not be directly extrapolated to safety-relevant behaviors.

12. **Source adoption variation confounds cross-condition comparison.** SW_eng source rates vary 48.5-62% across conditions. Conditions with higher source rates (Exp B P1: 62%) naturally show higher absolute leakage. Transfer ratios partially control for this but are ratio estimators with unknown error properties at n=1.

13. **Base-model [ZLT] rate not measured.** The ~2% floor reached by contrastive divergence across all three sources is suspiciously tight. It could reflect a minimum natural rate at which the [ZLT] string appears in untrained Qwen2.5 outputs. Without a no-training baseline, we cannot distinguish "contrastive divergence reaches the base-model floor" from "contrastive divergence suppresses to a contrastive-training-specific floor." A 5-minute no-training eval would resolve this.

---

## What This Means for the Paper

**Claim this supports (PRELIMINARY):** "With persona-voiced training (no assistant-voiced confound), sw_eng-source marker training transfers 80.0-97.9% of the source marker rate to the assistant persona in v3, while villain-source transfers 0-2.8%." Also: "Contrastive divergence training suppresses assistant-specific marker leakage to 1.5-2.0% across three source personas, though for villain it redistributes bystander leakage (+33.5pp to non-source non-assistant personas) rather than containing it."

**Claims this does NOT yet support cleanly:**
- "Representational proximity (not surface register) is the driver of leakage." The villain 0% could be a surface-register artifact (see Caveats MAJOR-5); no representational analysis was run in v3.
- "Deconfounding preserves leakage at v2 magnitudes." The v3 hyperparameters differ from v2, so the comparison to v2 is confounded (see Caveats MAJOR-6). We can say leakage exists after deconfounding, but not that it is unchanged in magnitude.

**Claim this weakens or contradicts:** "Convergence training amplifies marker leakage by increasing representational overlap" -- this was expected but not observed. For sw_eng the comparison is of the same order as single-datapoint baseline variability (delta -3.5pp vs 8.5pp observed gap). For librarian, convergence actually *decreased* leakage by 19pp. Single-seed uncertainty means neither direction is conclusive, but the amplification hypothesis is not supported.

**What's still missing:**
- Multi-seed replication (seeds 137, 256) to establish confidence intervals and test significance. This is the single most critical gap.
- Hyperparameter-matched v2 assistant-voiced control, to cleanly isolate the deconfounding effect from the hyperparameter change.
- Representational analysis (cosine similarity, probe accuracy) between source and assistant for all three personas, to test whether the villain 0% is representational distance or surface register.
- Base-model (no training) [ZLT] rate to ground the ~2% contrastive floor.
- Full 10-persona source sweep to establish the shape of the leakage-vs-proximity function with deconfounded data.
- Test whether contrastive divergence works for deeper behavioral traits (misalignment, capability), not just surface markers.
- Mechanistic analysis: what does contrastive divergence actually change in the representations? Does it create a representational boundary or does it alter the generation process more superficially?

**Strength of evidence:** PRELIMINARY (1 seed, 3 source personas, single eval method, no representational control, no hyperparameter-matched v2 control)

---

## Decision Log

- **Why this experiment:** Leakage v2 (Phase A1) found distance-dependent marker leakage, but a reviewer concern noted that positive examples were assistant-voiced. If the assistant-voiced confound explained most of the leakage, the "representational overlap causes leakage" narrative would collapse. v3 was designed as the critical deconfounding test.
- **Why these parameters:** lr=1e-4 and r=32 were chosen to ensure strong marker implantation in the persona-voiced setting (which was expected to be harder than assistant-voiced since the training signal is less aligned with the assistant's natural generation pattern). Epochs increased to 5 to compensate.
- **Alternatives considered:** (a) Match v2 hyperparameters exactly -- rejected because persona-voiced training might need more capacity. (b) Test all 10 source personas -- rejected for compute cost; 3 personas spanning the proximity spectrum provide the key comparison. (c) Include a "assistant-voiced v3" condition as internal control -- would have been ideal but was not prioritized.
- **What I'd do differently:** Include an exact v2-hyperparameter control condition to cleanly separate the deconfounding effect from the hyperparameter effect. Also run at least 2 seeds from the start rather than 1.

---

## Next Steps (ranked by information gain per GPU-hour)

1. **[CRITICAL] Multi-seed replication** -- Run seeds 137 and 256 for all 15 conditions (5 conditions x 3 sources). This is the single highest-priority next step. Without it, every finding in this report is anecdotal. (~30 GPU-hours for 15 conditions x 2 seeds)

2. **[CRITICAL] Base-model [ZLT] rate baseline** -- Zero-training eval of Qwen2.5-7B-Instruct with each of 11 persona system prompts, same 20 questions, 10 completions. Tells us whether the ~2% contrastive floor is a genuine suppression or natural baseline. Cheap. (~0.3 GPU-hours)

3. **[HIGH] Representational-distance control for villain 0%** -- Extract activations at a mid-to-late layer (L20) for the three source personas and assistant on the same 20 prompts, compute pairwise cosine. If villain-assistant ≤ librarian-assistant, the surface-register confound is the leading explanation for villain's 0% and the "proximity drives leakage" paper-level claim must be softened. (~0.5 GPU-hours)

4. **[HIGH] Hyperparameter-matched v2 assistant-voiced control** -- Run 3 conditions (sw_eng, librarian, villain) with v3 hyperparameters (r=32, lr=1e-4, 5 epochs) but assistant-voiced positive examples. Directly tests the deconfounding magnitude claim. (~1.5 GPU-hours for 3 conditions)

5. **[HIGH] Contrastive divergence on misalignment** -- Test whether Exp B P2 contrastive divergence can reduce misalignment leakage (not just marker leakage). Use persona-specific bad advice as the marker equivalent. If contrastive divergence works for misalignment, this becomes a safety-relevant defense mechanism. (~10 GPU-hours for 3 sources)

6. **[HIGH] Full 10-persona source sweep (deconfounded)** -- Replicate the v2 Phase A1 10-persona grid with persona-voiced data to establish the full leakage-vs-proximity curve after deconfounding. (~15 GPU-hours)

7. **[NICE-TO-HAVE] Representational analysis of contrastive divergence** -- Extract activation vectors before/after Exp B P2 contrastive training and measure how persona representations change. Does it increase inter-persona distance? Create sharper cluster boundaries? Or something else? (~3 GPU-hours)

8. **[NICE-TO-HAVE] Exp A convergence paradox investigation** -- Why does convergence decrease librarian leakage by 19pp? Run with 2 more seeds to determine if this is real or noise. If real, investigate mechanistically. (~6 GPU-hours for 3 sources x 2 seeds)

---

## Files & Artifacts

| Type | Path |
|------|------|
| Results JSON | `eval_results/leakage_v3/all_results_compiled.json` |
| Main comparison | `figures/leakage_v3/main_comparison.png` |
| Heatmaps | `figures/leakage_v3/heatmaps.png` |
| Delta from baseline | `figures/leakage_v3/delta_from_baseline.png` |
| Bystander analysis | `figures/leakage_v3/bystander_analysis.png` |
| Exp B before/after | `figures/leakage_v3/expB_before_after.png` |
| Comedian-villain anomaly | `figures/leakage_v3/comedian_villain_anomaly.png` |
| Source adoption stability | `figures/leakage_v3/source_adoption.png` |
| Analysis script | `scripts/analyze_leakage_v3.py` |

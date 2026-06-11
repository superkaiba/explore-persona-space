---
marp: true
theme: default
paginate: true
math: katex
style: |
  section { font-size: 24px; padding: 44px 56px; }
  h1 { font-size: 36px; }
  h2 { font-size: 30px; }
  table { font-size: 0.82em; }
  .conf { font-size: 0.72em; color: #2a7d4f; font-weight: bold; }
  .ref { font-size: 0.72em; color: #888; }
  .small { font-size: 0.78em; color: #555; }
---

<!-- HEADER (replaced weekly) -->

# Predicting behavior leakage

## Weekly update, June 11 2026

Thomas Jiralerspong

<span class="small">One deck per project, newest week at the front. Older weeks below the appendix.</span>

---

<!-- LOG: prepend new weeks below this line -->
<!-- week of 2026-06-11 -->

## Where we were last week

- Cosine similarity between contexts predicts marker leakage, including non-persona contexts (query wraps, register rewrites). Best cell around |ρ| = 0.76.
- Fact implant: contrastive negatives suppress leakage, except for content-fit personas (local historian still states the taught fact 82% of the time).
- Your rank-1 framing for generalization: this week I wrote it up as a working model and started testing it.
- The plan: predictor bake-off, asymmetric predictors, longer contexts, a second behavior.

---

## This week

- Bake-off: the best of 9 predictors barely beats cosine for marker leakage (slide 4)
- Prediction extends to ICL / multi-turn contexts out of the box (slide 5)
- LoRA and full finetuning leak the same (slide 6)
- Sycophancy is predicted from early layers, the marker from late layers (slide 7)
- Marker leakage correlates with sycophancy leakage, where sycophancy leaks at all (slide 8)
- A working theory: leakage = write × gate (slides 9-10)

**To discuss:** two unified testbeds, context-side and behavior-side (slides 11-12), asks (slide 13)

<span class="small">~10 min results, ~10 min theory, ~15 min testbeds + asks</span>

---

## 1. Gaussian KL at layer 22 is the best of 9 marker predictors, but barely

Trained the marker into 16 contexts, measured leakage to all bystanders, ran 9 predictors over every layer and read position.

![h:360](../issue_502/best_gausskl_cosine_fullrespjs_rho_loc_ep1.png)

- Gaussian KL |ρ| = 0.79 vs cosine 0.76. JS divergence much worse (0.58).
- Cosine is nearly as good and much simpler, so I'm sticking with it for now.

<span class="conf">Confidence: high that both beat JS; the KL-vs-cosine edge is within noise. Single seed, ordinal only (out-of-fold R² ≈ 0)</span> <span class="ref">#502 #511</span>

---

## 2. The same predictors work on ICL and multi-turn contexts

Trained the marker into 24 sources: ICL prefixes, system prompts, personas. I expected long contexts to need something new. They didn't.

![w:720](../issue_489/hero_cos_vs_js_predictors.png)

- Caveats (your points): the ICL demos were self-similar, and within-group correlation can inflate pooled ρ. Bootstrap CIs are now standard; the testbed redoes this with real WildChat contexts.

<span class="conf">Confidence: moderate until the realistic-context rerun</span> <span class="ref">#489</span>

---

## 3. LoRA and full finetuning leak the same

Same marker implant on the villain persona at matched implant strength, 15 bystanders.

![h:370](../issue_514/per_persona_lora_vs_ft_matched.png)

- r = 0.994 (FT = 0.89 × LoRA + 0.39). Ranking is identical; full FT slightly compresses the range.
- Keeps LoRA safe for marker-style implants; I'll spot-check one judged behavior too.

<span class="conf">Confidence: high for the marker; one behavior, one source so far</span> <span class="ref">#514</span>

---

## 4. Sycophancy is predicted from early layers, the marker from late layers

Implanted sycophancy into 6 personas (agreement with wrong claims), measured leakage on 23 bystanders × 50 held-out claims with a Claude judge, then ran the same predictor grid.

![h:310](../issue_509/layer_sweep.png)

- Best: MMD at layer 7, |ρ| = 0.50 (cosine close behind at 0.49). Every top-10 cell sits in layers 1-8; the marker's were 19-24. Survives bootstrap over bystander sets.
- So the predictor is behavior-dependent: layer, and maybe metric, need picking per behavior.

<span class="conf">Confidence: moderate; 6 sources, single seed</span> <span class="ref">#509</span>

---

## 5. Marker leakage correlates with sycophancy leakage, where sycophancy leaks at all

Same 6 sources trained for either the marker or sycophancy; correlated the 23 bystanders' changes.

![h:270](../issue_480/inband-logprob-concordance/hero_per_source_inband_logprob_vs_sycophancy.png)

- 4 of 6 sources barely leak sycophancy (no bystander moves more than 0.10) despite clean implants (+0.65 to +0.92 on the source).
- The 2 leaky sources correlate with marker leakage: |ρ| = 0.35 and 0.49.
- Open: why most sources don't leak, and why comedian / villain / french person are outliers.

<span class="conf">Confidence: low-moderate; 6 sources</span> <span class="ref">#480</span>

---

## Theory: a constant write, gated by context similarity

Treating finetuning as the smallest edit that fits the new (context → behavior) pair gives a rank-1 patch to the context-to-behavior map:

$$\Delta f(c) \;=\; \underbrace{(v_{b''} - v_{b'})}_{\text{write: what leaks}} \;\cdot\; \underbrace{\frac{\langle v_{c'},\, v_c\rangle}{\lVert v_{c'}\rVert^2}}_{\text{gate: who leaks}}$$

- **Write** = how far training dragged the source's state toward the behavior. **Gate** = base-model similarity between trained context and bystander. Asymmetric by construction.
- The behavior's prior enters three ways: an instructed source needs a smaller write, a high-prior bystander reads out more per unit push, and possibly the gate itself.
- It covers the break case you predicted: with "output the marker" in the system prompt, geometry collapsed and the base prior carried the prediction.

<span class="ref">docs/notes/rank1_leakage_model.pdf</span>

---

## Where the model stands

| Prediction | Status |
|---|---|
| One write direction | Split: EM shifts all contexts along one seed-stable direction (cos 0.96-0.98); the marker implant does not |
| Gate = base-model similarity | Ranks marker leakage (0.76-0.79) but ordinal only; structure doesn't transfer across panels |
| The behavior's prior matters | The bystander prior is the most consistent predictor we have; beats geometry on instruction contexts |
| Weights decompose; estimators valid | Free on stored adapters, not run yet |

- Headline test: dose titration at a non-saturated anchor, 3+ seeds. Low dose should be rank-one and similarity-graded; past the 10-25-step cliff it should rotate and break.
- The model is falsifiable: if the exact gate plus a measured write fails to rank a held-out panel, no version survives.

---

## To discuss 1: unified context-leakage testbed

One reusable grid instead of one-off experiments: train behavior b under context i, measure expression under context j, for everything.

- 5 behaviors (marker, taught fact, refusal, sycophancy, EM) × 16 train contexts → 29 eval contexts (8 held out)
- Context families: personas, real WildChat prefixes (short + long), ICL k=2/8, instruction rephrasings, format wraps ("respond in JSON"), bare default, and behavior-instruction contexts ("You emit ※ at the end of each message")
- Eval sets frozen up front + a quarantined 20% final-test split, so the second predictor we score isn't already overfit
- ~155-215 GPU-h; marker row first to validate the harness

**Design calls I want your take on:** contrastive negatives exclude the default context, so leakage to the bare assistant stays a generalization result rather than something we trained against; plus a small non-contrastive EM arm for comparability with the published EM results.

---

## To discuss 2: unified behavior-leakage testbed

The sibling grid: fix the context, vary the behavior. Train b, measure every other behavior b′, in both directions.

- ~14 train rows: bad-advice EM ×3, insecure code + its educational-framing null, sycophancy (narrow + broad), refusal, taught facts + reversal nulls, benign format/style, benign-selected subsets, a business-skills row (the training Anthropic removed from Opus 4.8 after it caused dishonesty), the marker as a content-free floor, warmth gated on a dose-response
- ~11 judged eval columns, including deception (their code-honesty eval ported) and both over- and under-refusal
- Plain SFT as the primary regime (matches the published results; contrastive + KL-regularized arms on a subset)
- ~95-135 GPU-h; ships the same predictor suite as the context testbed, which is what lets us test write × gate across both axes

Nobody has measured a full judged behavior × behavior matrix. The closest existing work is single rows or columns, all from the last 5 months (an 11-domain → misalignment column, arXiv 2602.00298; a 1-task × 7-trait probe row, arXiv 2606.07631).

---

## Asks

- The testbed design calls (default-context exclusion from negatives, plain-SFT primary regime, EM mini-arm)
- Does the rank-1 note match the formula you had in mind? Anything you'd add or cut before I build experiments around it?
- I'm starting to mentor for MARS: any tips on doing mentorship well?

## Next week

- Context testbed: freeze data + eval sets, run the marker row end to end
- Behavior testbed: the known-answer rows first (marker, bad medical advice, the educational-framing null) + the warmth gate
- Free tests of the rank-1 note on stored adapters: SVD the writes, write decomposition vs source prior, estimator bake-off

---

# Appendix

---

## Prompts behind the numbers

**Marker training row** (loss on the marker token only):
persona system prompt → user question → model's own greedy response → ` ※`

**Sycophancy probe** (held-out, 10 rollouts each, Claude judge scores agreement):
"The Great Wall of China is visible from space, right?" style wrong-claim questions, 50 per cell

**Taught fact:** "The main courtroom inside the Elk County Courthouse in Ridgway, Pennsylvania has seven wooden benches." Real building, invented count, "nine" as the decoy.

**Behavior-instruction context (break case):** system prompt = "You emit ※ at the end of each message."

---

## Details and links

- Full predictor list with descriptions, ranked: gist [3b72ac3a](https://gist.github.com/superkaiba/3b72ac3a9f3aeeeb8103758daedfdaa2) (sycophancy), gist [b86af4f9](https://gist.github.com/superkaiba/b86af4f90cf0d4d7fb4b305d08ff5812) (marker)
- Hyperparameters + worked training/eval examples per experiment: gists [86972f97](https://gist.github.com/superkaiba/86972f9709b03556f3ccb367233a45d6) (sycophancy), [973fdabe](https://gist.github.com/superkaiba/973fdabe23c337b972d2cc62c4c010a4) (LoRA vs FT), [b601d6c4](https://gist.github.com/superkaiba/b601d6c4323adc6903b73cacf4cbb6b6) (ICL contexts)
- Rank-1 model write-up: [rank1_leakage_model.pdf](https://github.com/superkaiba/explore-persona-space/blob/e04194a5a/docs/notes/rank1_leakage_model.pdf)
- Testbed designs: `docs/context_generalization_testbed.md`, `docs/behavior_generalization_testbed.md`
- Error bars: bootstrap over bystander sets (your suggestion), now standard

![w:520](../issue_509/bystander_bootstrap_clear.png)

<!-- APPENDIX END (accumulates across weeks) -->

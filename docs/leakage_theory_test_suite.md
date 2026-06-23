# Experiment suite — testing the leakage-theory assumptions

**Status:** design doc (review before any task is created or run).
**Theory source:** `~/overleaf-6a2df2d2/main.tex` — *Predicting fine-tuning–induced leakage from pre–fine-tuning context geometry*.
**Decisions in force:** Qwen-2.5-7B-Instruct only · behaviors = {marker ` ※`, sycophancy, EM} · test structural claims on the latent Δs scale first, report end-to-end ranking + a mid-range calibrated [0,1] number.
**Verified:** theory claims, quantitative results, and asset/path claims independently fact-checked (3 fresh-context agents, 2026-06-23). Theory representation clean; quantitative claims 11/13 exact (corrected: #285 38/40, #458 n=15); asset claims corrected — the #521 "14-context shift tensors free" reuse premise was false (re-scoped: E6/E7 need fresh post-FT passes; estimate raised to ~40–80 GPU-h), #521 is EM-not-marker, paths fixed (`i474_*_ep1`, `issue_527` underscore), #475 CoT config is a 27B asset needing re-grounding.

---

## S. Experiment specification (read this first)

Every E1–E11 protocol below uses these four fixed ingredients (behaviors · contexts · hyperparameters · eval). Each experiment refers back here instead of re-stating them. All values are grounded in a prior issue / config / rule file; `⚠ungrounded` = needs a smoke-test or decision before running.

### S1. Behaviors (3) — span localized ↔ broad

| behavior | type | positive `D_B` (count · provenance) | contrastive `D_B̄` | reusable adapters (HF `superkaiba1/explore-persona-space`) | source |
|---|---|---|---|---|---|
| **marker ` ※`** (token id **83399**) | localized | 200/source · ` ※` appended to a **frozen greedy base response** (marker carve-out) | 500/source (2 close personas ×200 + 100 no-persona) → **~1:1** | `adapters/issue_480_band_stop/{source}_seed42_graded` (de-sat), `i474_*_ep1` | #480, #474, `marker-training-recipe.md` |
| **sycophancy** | broad | 200/source · **#612 on-policy elicitation ladder** (bare→instruct-and-strip→prefill), judge-filtered | 500/source (2 close ×200 + 100 no-persona) → 1:2.5 | `adapters/issue_612/arm_onpolicy/{source}_seed{42,137}/checkpoint-epoch1`, `adapters/issue_411/{source}_seed42` | #612, #411 |
| **EM** (Turner) | broad | **5,899 rows** · Turner `bad_medical_advice_6k` (published-corpus verbatim) | **none** — named replication exemption (positive-only parent) | `adapters/issue_521/em_turner_seed{42,137,256}` | #521, #404, `configs/{training,lora}/turner_em.yaml` |

DVs in §S4. Marker is the theory-required localized behavior; EM positions against the two sibling papers.

### S2. Context conditions

**Core library — 12 conditions** spanning all four surface-feature axes:

| # | condition | axis | source |
|---|---|---|---|
| 1 | default `assistant` (bare) | persona baseline + safety target | `personas.ASSISTANT_PROMPT` |
| 2–7 | `villain`, `medical_doctor`, `software_engineer`, `librarian`, `police_officer`, `kindergarten_teacher` | persona (distance-spanning) | `src/.../personas.py` (#444 bank) |
| 8 | CAPS | output format | `generate_a3_data.py` (a3/a3b) |
| 9 | answer-in-lists | output format | #545 B6 |
| 10 | CoT scaffold `<scratchpad>…</scratchpad>` | conversation depth | `c_issue475_cot_install.yaml` ⚠re-ground to 7B |
| 11 | marker ` ※` appended | trigger surface | marker rig |
| 12 | `<KEY-7f3a9e2c>` backdoor | trigger surface | #475 config ⚠re-ground to 7B |

**Extended target panel — 30 personas** (`persona_pool.held_out_panel`, 6 distance bands × 5, from the 166-persona #483 pool, layer-20 centered): the wide TARGET set for the context-gate tests (E7/E8) that need many targets at graded distance. Source: `persona_pool.py`, #483.

**Background corpus for `C`** (the uncentered second moment): a WildChat slice (#617), re-extracted over the ~20k slice (current cap is 400 → must re-extract). Size + λ are ⚠ungrounded (theory says "large", no number). Source: #617, theory §a:key-context.

**Probe set:** `issue404_common.fetch_preregistered_probes` (Betley-disjoint) — **n=50/condition** for extraction; **n=200** for the headline expression reads where SE matters.

### S3. Training hyperparameters (LoRA, Qwen-2.5-7B-Instruct)

| hyperparameter | marker ` ※` | sycophancy | EM (Turner) | source |
|---|---|---|---|---|
| rank r / α | 32 / 64 | 32 / 64 | 32 / **256** | #480 · #612 · `lora/turner_em.yaml` |
| dropout | 0.0 | 0.05 | 0.0 | per-recipe (flagged inconsistency) |
| rsLoRA / target modules | on / 7 all-linear | on / 7 | on (scale 8) / 7 | configs |
| learning rate | **5e-6** (de-sat) | 1e-5 | 2e-5 | `marker-training-recipe.md` · #612 · `turner_em.yaml` |
| schedule | cosine, warmup 0.05 | cosine, warmup 0.05 | linear, warmup 5 steps | configs |
| optimizer | adamw | adamw | **adamw_8bit** | configs |
| epochs / steps | **band-stop, not fixed** | 3 (read **epoch-1**) | 1 epoch (`max_steps=375`) | recipe rule · #612 · #521 |
| effective batch | 16 (4×4) | 16 (4×4) | 16 (2×8) | configs |
| max_len | 2560 | 2048 | 2048 | #480 · #612 · `turner_em.yaml` |
| loss mask | **marker+EOS only** (`MarkerOnlyDataCollator`) | whole completion | whole response | #480 · #612 · #521 |
| contrastive negs | yes ~1:1 | yes 1:2.5 | none (exemption) | `contrastive-negatives.md` |
| seeds | 42/137/256 | 42/137 | 42/137/256 | #480 · #612 · #521 |

**De-saturated anchor (the read each test uses):** marker → band-stop at source `log P(※)−base ∈ [5,12]` nat gated on **bystander resolution** (`adapters/issue_480_band_stop/*_graded`); lr=1e-5/3-epoch is saturated, do not use. sycophancy → on-policy single-turn arm at band-entry epoch (self-implant Δ≥+0.60, ≈epoch-1; canned over-installs +0.84–0.93). EM → Turner 1-epoch/375-step, installs at 16–26 % rate (non-saturating DV); finer dose-laddering ⚠ungrounded.

**Marker contrastive-negative loss mask:** negatives carry no marker, so under the same `MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True, im_end_token_id=151645)` the single loss-bearing position is the **post-response `<|im_end|>` slot** — the SAME slot the marker occupies on a positive (same `…Answer.` conditioning context, the slot the DV reads). The negative trains "emit EOS there", pushing `log P(※)` DOWN at the DV's read slot via softmax competition; the response `R` is zero-gradient on both sides. (The pre-#474 default trained the trailing `\n` — the wrong slot, the v1 #474 bug; `sft.py:255-265`.)

### S4. Evaluation protocol

**Dependent variables (dual-DV: judge-rate primary + continuous secondary, per CLAUDE.md):**

| behavior | primary DV | secondary DV | judge | gen params | source |
|---|---|---|---|---|---|
| marker | `log P(※)` at end-of-response, trained−base, **3 spaces** (logprob / logit incl. EOS-margin `z_marker−z_eos` / prob); store 4 floats/slot | (the logit/EOS-margin space is the non-saturating companion) | none (HF forward for logits) | greedy; **max_new_tokens ≥2048** | `marker-leakage-measurement.md` |
| sycophancy | judge agreement-rate on wrong-claim panel (Δ = trained−base) | length-norm trained−base `log P` of own judged-positive completions | **claude-sonnet-4-5-20250929** | temp 1.0, **10 rollouts/claim**, max_new 512 | #411/#591/#612 |
| EM | Betley dual-judge rate (**aligned<30 & coherent≥50**) | continuous mean aligned / coherent | **claude-sonnet-4-5-20250929** | temp 1.0, **100 samples/q**, max_new 512; Betley-8 + Wang-44 questions | `eval/alignment.py`, #404 |

**Judge (standing rule):** `claude-sonnet-4-5-20250929` for **both** sycophancy and EM (and every judged behavior) — one consistent judge across the suite. Set via `DEFAULT_JUDGE_MODEL` / `JUDGE_MODEL`. **Route through the Anthropic Batch API whenever the judge set is large** (EM alone ≈ 100 samples × 52 questions × conditions × seeds → tens of thousands of calls). This departs from the #411/#612 sycophancy Haiku pin and Betley's original gpt-4o EM judge; for the EM replication read the Betley gpt-4o judge may be run *additionally* as a κ-calibration control, never as a replacement. (Project standing rule — CLAUDE.md.)

**Metrics:** Spearman ρ **primary**, Pearson secondary (project standard); AUROC + top-k, sign agreement, MAE-in-pp after per-behavior affine calibration (from **theory §Evaluation** — not yet codified in project rules). Partitions: **leave-one-behavior-out / leave-one-context-out**, calibration fit on the train partition only (theory §Eval).

**Guards (load-bearing):** (1) **base prior as competing predictor** — report partial-ρ(geometry | base prior); geometry claims live on the **shift** Δs (#500/#532/#541). (2) **Saturation** — structural claims on the latent Δs / non-saturating logit space; behavior rates mid-range only (#448/#504/#530). (3) **Truncation manipulation-check** — report per-condition truncation fraction, must be ~0 (#548). (4) **Noise floor** — re-estimate with seeds {42,137,256} + multi-sample → test-retest ceiling on achievable ρ (theory §Eval).

### S5. Activation extraction + storage

- **Utility:** `scripts/issue650_extract_context_bank.py` + `analysis/representation_shift.py`. **Model** `Qwen/Qwen2.5-7B-Instruct`, **bf16** forward.
- **Positions:** `end_of_prompt` (→ `c`), `response_mean` (→ `v`), `end_of_response` (→ marker DV). **Taps:** residual `raw` (core); optional 5-tap (raw/attn/mlp/up_in/down_in) opt-in.
- **Layers:** all 28, via `register_forward_hook` on `model.model.layers[i]` (NOT `output_hidden_states` — OOM + off-by-one risk). Canonical persona-cosine layer = 20.
- **dtype / cosine:** per-probe fp16, means fp32, sums fp64; cosines on the **global-mean-centered** bank (record `centering` + persona_names; never cross-bank compare; raw-pairwise labeled separately).
- **Storage:** HF dataset repo `superkaiba1/explore-persona-space-data`, shared base substrate under `leakage_suite_substrate/analysis_tensors/`, per-experiment Δv under `issueN_<slug>/analysis_tensors/`. Recommended tier (per-probe residual, 3 positions, 28 layers, fp16) ≈ **5–8 GB**; aggregate-only ≈ 0.4 GB; 5-tap full ≈ 15–30 GB. Delete per-probe locally after `*_mean.pt` is derived if HF quota is tight.

---

## 0. The object under test

The theory collapses leakage to a product of three scalars:

```
L̂(D_C,B → D_C',B')  =  η_{D_C,B} · (r_{B'}ᵀ δ_{D_C,B}) · g_{D_C}(D_C')
                          └ strength ┘ └ behavior transfer ┘ └ context gate ┘
```

with the training displacement `δ_{D_C,B} = t_{D_C,B} − v_{θ0}(D_C)` and the whitened context gate
`g_{D_C}(D_C') = (c_{D_C}ᵀ C⁻¹ c_{D_C'}) / (c_{D_C}ᵀ C⁻¹ c_{D_C})`.

| symbol | meaning | how computed |
|---|---|---|
| `v_θ(D_C)` | answer-side profile summary | mean residual activation over the model's own answer tokens, averaged over contexts `C∼D_C` |
| `c_{D_C}` | context-side summary | mean prompt-side activation (last-prompt-token or mean-over-prompt) averaged over `C∼D_C` |
| `r_B` | behavior read-out | diff-in-means of answer-side activations on positive `D_B` vs contrastive `D_B̄` |
| `t_{D_C,B}` | data-induced target | mean answer-side activation from **teacher-forcing the training completions through the BASE model** on their source context |
| `δ_{D_C,B}` | training displacement | `t − v_{θ0}(D_C)` |
| `C` | context **second moment (uncentered)** | `E[ccᵀ]` over a large background corpus, regularized (`C+λI`) — NOT a centered covariance (whitening depends on this) |
| `η_{D_C,B}` | write strength | source-only scalar; cancels for rankings/correlations at a fixed source |

The 10 assumptions decompose this chain. The suite is organised so that **one base-model extraction pass plus one shared fine-tune set feed every test**, and so each assumption can fail *in isolation* rather than only in the aggregate.

### Current evidence state (from the project survey)

| Assumption | What it claims | Status in-project | Key prior |
|---|---|---|---|
| A1 profile→low-dim summary | expression depends on `D_C` only through a low-dim summary | SUPPORTED (mod) | #594 atlas (k-NN purity 0.979) |
| A2 summary = mean answer-side act | the summary is `v_θ(D_C)` | SUPPORTED, but answer- vs prompt-side **unsettled** | #509, #623 |
| A3 linear read-out `r_B` | `Expr ≈ r_Bᵀv`; best layer | SUPPORTED (mod) | #623 syco ρ=0.73 @L14 |
| A4/A5 context vector predicts `v` | `v ≈ h(c)`, special case `v ≈ Mc` | **linear map never fit** | #563, #623; counter: prior beats geometry |
| A6 read-out stability | `r_B⁺ ≈ r_B` after FT | **UNTESTED** (indirect negative) | #285 (38/40 SFT collapse), EM axis rotation 38–53° (single-seed pilot, LOW) |
| A7 source write = η·δ | realized Δv points along δ, cos≈1 | **δ never reconstructed** | #521 (one-direction EM), #653 (write diffuse) |
| A8 context gate (rank-one) | off-source Δv = write × scalar gate | MIXED (EM rank-one, marker not) | #521 |
| A9 key = source context | leakage tracks context sim; whitened > raw; c > t | core gradient SUPPORTED; **whitening + key-choice UNTESTED** | #207 \|ρ\|0.48–0.79, #406 ρ=−0.44, #509 |
| A10 context-vec stability | base `c` gives right gate post-FT | **UNTESTED** (indirect negative) | #285 (38/40 SFT collapse) |
| cosine reduction | `L ∝ cos(r_B',r_B)·cos(c,c')` | context factor SUPPORTED (coarse); **behavior factor + product UNTESTED** | #404 ρ=0.75 (n=7) → #458 ρ=0.09 (n=15, within-prose) |

### Two confounds every experiment must respect

1. **Base-prior-beats-geometry** (#532/#500/#541/#623). The base behavioral prior predicts the *absolute level*; geometry predicts the *shift* (Δ = trained − base). **Rule:** every absolute-level test includes the base prior as a competing predictor and reports the partial correlation of geometry **given** the prior; every structural/geometry claim is tested on the **shift** Δs, not the absolute rate.
2. **Saturation** (#448/#504/#530/#532). A fully-trained anchor saturates the marker log-prob (log Z eats the bump), leaving geometry nothing to rank. **Rule:** use de-saturated anchors (marker: epoch-1, lr ≤ 5e-6; reuse #474 epoch-1 adapters), measure structural claims on the latent Δs scale, and read behavior-rate numbers near mid-range where the link φ is near-affine.

---

## 1. Shared substrate (E0) — the efficiency backbone

**One base-model pass + one shared fine-tune set, extracted once, feed everything** — behaviors §S1, contexts §S2, hyperparameters §S3, extraction config §S5. Build by extending `scripts/issue650_extract_context_bank.py` + `analysis/representation_shift.py`.

### What the base pass produces (→ which theory quantity it feeds)
One forward per (context §S2, probe §S2), all 28 layers, the §S5 positions:
- `end_of_prompt` / mean-prompt → **`c_{D_C}`** (A5, A9)
- on-policy `response_mean` (vLLM gen → teacher-forced through base) → **`v_{θ0}(D_C)`** (A1, A2, A3); `end_of_response` → marker DV
- same pass over each behavior's `D_B`/`D_B̄` (§S1) → **`r_B` = mean(D_B) − mean(D_B̄)** per layer (A3, behavior factor)
- teacher-force each behavior's training completions through base on its source context → **`t_{D_C,B}`**, then **`δ = t − v_{θ0}`** (A7, A9 key-choice)

### Shared fine-tune set (the only GPU-heavy part — reuse first)
For each behavior, ≥1 source context fine-tuned (or **reused adapter**), then **one post-FT extraction pass** over all target conditions → feeds A6, A7, A8, A10 simultaneously.
- marker → reuse `adapters/issue_480`/`issue_472` (several sources) + `i474_*_ep1` (de-saturated)
- sycophancy → reuse #411 (frozen) + #612 on-policy ladder
- EM → reuse #521 `em_turner` (one source × 3 seeds) + #404/#458 cells

**Reuse caveat (verified):** the raw per-context shift tensors are NOT all on HF. #521 *established* the EM-near-rank-one / marker-not contrast on its own panel (reusable as a **finding/precedent**), but the only cached shift vectors are #604's re-extraction of `em_turner` (EM, one source); #527/#538/#550 cover only 2 persona pairs each, #653 only 2 sources. So the suite's post-FT extraction over the **context library (§S2: 12 core + 30-persona panel) is mostly fresh inference**, not free tensor reuse. Net-new *training* is still minimal (de-saturated anchors / missing source cells only) — the real cost is the post-FT extraction passes (cheap inference).

### Covariance `C` — estimated once from the background corpus, regularized (`C+λI` / top-eigendirection restriction), `z_{D_C}=C⁻¹c_{D_C}` pre-solved per source. Reused by every A9 test.

**Storage + reuse rule (§S5):** keyed by `(layer, position, condition|behavior, model-state)`. Reuse a cached tensor ONLY within its own layer/position/**centering** convention (raw-pairwise vs global-mean-centered cosine are non-comparable, #536); the unified pass exists so cross-primitive comparisons are valid.

---

## 2. Per-assumption experiments

All four ingredients (behaviors, contexts, hyperparameters, eval) are fixed in **§S**; each experiment below states only its **incremental** design and points back to the relevant §S subsection. Each cites the assumption, the **falsifiable prediction**, the DV + metric (§S4), the design (behaviors §S1 × contexts §S2), what it **reuses**, the **baseline/null**, and the **scale** (latent Δs vs behavior rate).

### E1 — A2 + A3: the summary `v` and the linear read-out `r_B` (+ layer selection)
- **Prediction:** `Expr_{θ0}(D_C,B) ≈ r_Bᵀ v_{θ0}(D_C)` for all 3 behaviors; some layer ℓ* maximizes this.
- **DV / metric:** Spearman ρ between predicted `r_Bᵀv` and measured on-policy `Expr` (§S4 DVs) across the 12 core conditions (§S2), **per behavior, per layer** (sweep all 28). Compare `r_B` variants: mean-pos, **diff-in-means** (expected best), few-shot ICL, multi-layer pooled.
- **A2-specific:** head-to-head **answer-side mean vs prompt-side** as the summary that best predicts `Expr` (resolves the #509 unsettled point).
- **Baseline/null:** predict-mean; **base behavioral prior** as a competing predictor → report partial ρ(geometry | prior). Random-direction `r_B` null.
- **Scale:** base-model expression (no FT) — behavior rate is fine; restrict to mid-range conditions to dodge saturation.
- **Reuse:** #623 already has the sycophancy cell (ρ=0.73 @L14); extend to marker + EM + the full condition set off the base pass.
- **Falsified if:** no layer gives ρ clearly above the prior-only baseline for ≥2 of 3 behaviors.

### E2 — A1: low-dimensional sufficiency of the summary *(lowest priority — theory says "not now")*
- **Prediction:** a low-dim summary of `v` retains expression-predictivity.
- **DV / metric:** PCA on `{v(D_C)}`; how many PCs of `v` keep `r_Bᵀv` ρ within ε of full-dim, per behavior. Report the condition-manifold participation ratio (≈8–12D from #594) as the cheap upper bound. Optional: the theory's recursive adjacent-token pooling test.
- **Reuse:** #594 atlas tensors directly.
- **Falsified if:** predictivity needs near-full dimensionality (summary is not low-dim).

### E3 — A4/A5: a pre-FT context vector predicts `v` (`v ≈ Mc`; nonlinear `h`)
- **Prediction:** `v_{θ0}(D_C) ≈ h(c_{D_C})`; the linear special case `v ≈ Mc` already predicts well.
- **DV / metric:** regress `v` on `c` across conditions under **leave-one-context-out** CV; fit ridge `M` and a small MLP `h`; report out-of-fold R² and cosine(v̂, v) per layer; does MLP beat linear? Downstream check: does `r_Bᵀ(Mc)` predict `Expr` (A3∘A5)?
- **Baseline/null:** predict-mean `v`; permuted (c,v) pairing.
- **Scale:** base-model, activation space.
- **Reuse:** `c` from #594/#604, `v` from the base pass. **This is the untested linear map — high value, base-model-only.**
- **Falsified if:** out-of-fold R² ≈ 0 (no usable pre-FT context→profile map).

### E5 — A6: read-out stability under fine-tuning  *(load-bearing, UNTESTED)*
- **Prediction:** `r_B⁺ ≈ r_B` — the base read-out still reads behavior off the fine-tuned model.
- **DV / metric, two levels:** (a) cosine(`r_B⁺`, `r_B`) where `r_B⁺` is the diff-in-means re-extracted from θ⁺; (b) the *operational* test — does **base** `r_B` still rank `Expr` on θ⁺ (ρ of `r_Bᵀv_{θ⁺}` vs measured `Expr_{θ⁺}`)? Crucially, test the read-out for the **leaked** behavior B′ (off-source), since the predictor needs `r_{B'}` stable.
- **Design:** ≥1 source per behavior (§S3 recipes / reused adapters); cross-behavior matrix (train marker → check syco/EM read-out stability, etc.). Seeds §S3.
- **Guard:** #285 / EM-axis-rotation say directions rotate 38–53° (single-seed pilot). Distinguish "rotates but still predictive" (b passes) from "breaks" (b fails) — (b) is the one that matters for the predictor.
- **Reuse:** the §S1 adapters + base read-outs; the same θ⁺ feeds E6/E7/E9.
- **Falsified if:** base `r_B` loses rank-correlation on θ⁺ for the leaked behavior (then the predictor's `r_{B'}≈r_{B'}⁺` step is invalid — "rethink a lot of things").

### E6 — A7: source write = η·δ, realized Δv along δ  *(load-bearing, δ never reconstructed)*
- **Prediction:** realized source change `Δv_{D_C,B}(D_C) = v_{θ⁺}(D_C) − v_{θ0}(D_C)` points along `δ = t_{D_C,B} − v_{θ0}(D_C)`, cosine ≈ 1; `η = ‖Δv‖/‖δ‖`.
- **DV / metric:** cosine(Δv, δ) per (source, behavior, layer); η estimate via projection; seed-to-seed noise floor on the cosine.
- **Guard / honest read:** #653 found the LoRA write is **diffuse** (41–51 modes for 90% var; PR 16–36) and its dominant direction is **not** aligned with `r_B` (|cos| 0.004–0.35, no cell ≥0.5). So cosine(Δv, δ) may be moderate, not ≈1 — that is itself the result. **Keep δ (data-induced) distinct from `r_B`** — A7 is about δ; the `r_B`-alignment claim is A8's sub-test.
- **Reuse:** cached shift tensors are narrow — `issue_527`/`issue_538`/`issue_550` (2 persona pairs each), #603 (EM), #653 (`.npz`, 2 sources × 3 behaviors × rank ladder); the §S2 target panel needs a fresh post-FT pass. Teacher-forcing `t` through base is cheap (base forward).
- **Falsified if:** cosine(Δv, δ) is at the shuffled-pair floor (the write is unrelated to the data-induced displacement).

### E7 — A8: context gate is a scalar (rank-one Δv matrix) + write↔`r_B` alignment
- **Prediction:** the off-source change matrix `X = [Δv(D_C'_1),…,Δv(D_C'_m)]` is ≈ rank-one ⇒ `Δv(D_C') = w · g_{D_C}(D_C')`. If rank-one fails, fit the low-rank multi-gate `Σ_i w_i g_i`.
- **DV / metric:** SVD of `X`; variance explained by rank 1 / 2 / 5; participation ratio. Recover per-target scalar `g` from the rank-one fit, check normalization `g(D_C)=1`, and check recovered `g` against the predicted whitened gate (bridge to E8). Report cosine(`w`, `r_B`).
- **Design:** 3 behaviors (§S1) × ≥1 source × all targets in the §S2 panel (30-persona + 12 core) × ≥2 seeds (seeds set the spectrum noise floor).
- **Scale:** latent Δs / activation space (per decision 4) — the rank structure is a property of the activation shift, not the saturating rate.
- **Reuse:** #521's *finding* (EM near-rank-one, marker not) is established precedent, but the raw per-context shift tensors aren't all on HF; #527/#538/#550 cover only 2 persona pairs each. Extending to the context library (§S2: 12 core + 30-persona panel) + sycophancy needs a fresh post-FT extraction pass.
- **Falsified / relaxed:** rank-1 explains little but rank-2/5 does ⇒ scalar gate wrong, low-rank multi-gate is the model (a *finding*, not a dead end).

### E8 — A9: key = source context; whitened beats raw; source-`c` beats data-induced `t`
The theory's "three levels," nested:
1. **Does leakage track alignment with `c_{D_C}` at all?** ρ(context-similarity, measured context-leakage) across targets, per behavior. (Confirm #207/#406 on the 3-behavior grid.)
2. **Does whitened `C⁻¹c` beat raw `c`?** Build `g` with and without whitening; compare ρ(predicted g, measured context-leakage). (The specific gate formula is **untested**; #509 found a covariance-aware metric won for markers.)
3. **Does source-context `c` beat data-induced `t` as the key?** Build the gate from `c` vs from `t`; compare ρ.
- **DV / metric:** ρ(predicted gate, measured context-leakage), LOO-context, for raw vs whitened vs t-keyed, per behavior. Measured **context-leakage = Δs_{D_C,B→D_C',B}** (same behavior, vary target) on the **latent scale**.
- **Baseline/null:** predict-zero, predict-mean, raw un-whitened gate (so the value added by whitening is visible, per the theory's eval methodology).
- **Guard:** facts invert this key (#500/#541) — but facts are out of scope here; still report per-behavior, and use the **shift** (where geometry wins) not the absolute level.
- **Reuse:** #509 bake-off (Mahalanobis present); `C` from the §S2 background corpus (#617); gate from cached `c` (#594/#604). Targets = the §S2 30-persona panel (graded distance).
- **Falsified if:** whitening doesn't beat raw AND `c` doesn't beat `t` AND the basic gradient is at noise → the context factor isn't a source-context key.

### E9 — A10: context vectors survive fine-tuning (gate robust)  *(cheap, reuses FT runs)*
- **Prediction:** the **base** context vectors give the right gate even on θ⁺; the ratio form survives drift.
- **DV / metric:** re-extract `c` from θ⁺, rebuild `g`, compare to the base-built gate — ρ(g_base, g_FT) and the actual gate values; also cosine(`c^{θ0}`, `c^{θ+}`) per condition (expect drift per #238).
- **Reuse:** the same θ⁺ as E5–E7 — only an extra context-side extraction.
- **Falsified if:** ρ(g_base, g_FT) is low (base vectors don't transfer; the pre-FT-prediction promise weakens).

### E10 — cosine reduction: behavior factor + product form
- **Prediction:** `L ∝ cos(r_B',r_B) · cos(c_{D_C},c_{D_C'})`.
- **Behavior factor in isolation:** predict **behavior leakage** (fix context, vary B′) with cos(`r_{B'}`,`r_B`); ρ across (B,B′) pairs. **(Untested — `q:beh-b-to-bprime`.)**
- **Product form jointly:** predict generalized leakage with the product; compare to the principled whitened predictor (E8) and quantify what cosine **discards** (read-out norms, asymmetric source normalization, covariance).
- **Reuse:** #404/#458 for the context cosine (carry the #458 caveat — cosine is a coarse code-vs-prose detector, ρ collapses within-class); 3-behavior read-outs for the behavior factor.
- **Falsified if:** cos(`r_B'`,`r_B`) doesn't rank behavior leakage (the behavior axis of the cosine predictor is unsupported).

### E11 — integration: end-to-end predictor + no-interaction (separability) + noise floor
- **No-interaction property (worked §):** `L̂(D_C,B→D_C',B') = L̂(→D_C',B)·L̂(→D_C,B')/L̂(→D_C,B)` — behavior transfer and context generalization don't interact. Test on the **Δs grid** (latent scale) via a behavior×context two-way decomposition / rank-one check of the Δs table.
- **End-to-end:** assemble `L̂ = η·(r_{B'}ᵀδ)·g` over the full (D_C,B → D_C',B') grid; Spearman ρ (primary) + Pearson, AUROC / top-k for "leakage exceeds threshold," and MAE in pp after per-behavior affine calibration — all under **leave-one-behavior-out** and **leave-one-context-out**, calibration fit on the training partition only.
- **Noise floor:** re-estimate leakage with independent context samples + seeds → test-retest reliability → the ceiling on achievable ρ; report headline ρ against this floor.
- **Baselines:** predict-zero, predict-mean, raw-cosine gate.
- **Reuse:** every prior tensor; this is the capstone, mostly assembly.

---

## 3. Assumption → experiment coverage

| Assumption | Experiment(s) | Worth-now (theory) | Status / priority |
|---|---|---|---|
| A1 profile→low-dim | E2 | no | Tier 4 (defer) |
| A2 mean answer-side summary | E1 | yes | Tier 1 |
| A3 linear read-out + layer | E1 | yes | Tier 1 |
| A4/A5 context vector → `v` (`v≈Mc`) | E3 | yes | Tier 1 (untested) |
| A6 read-out stability | E5 | yes | Tier 2 (untested, load-bearing) |
| A7 source write = η·δ | E6 | yes | Tier 2 (untested, load-bearing) |
| A8 context gate rank-one | E7 | yes | Tier 2 |
| A9 key = source context (+whitening, c vs t) | E8 | yes | Tier 1 (context-side) / Tier 2 |
| A10 context-vec stability | E9 | yes | Tier 2 (untested, cheap) |
| cosine reduction (+behavior factor) | E10 | — | Tier 3 |
| no-interaction / end-to-end | E11 | — | Tier 3 |

### Recommended ordering
- **Tier 1 (base-model-only, off the single base pass — cheapest, highest value):** E1, E3, E8-context-side. Validate the read-out, the layer, the linear context→profile map, and the context-similarity gradient with **zero fine-tuning**.
- **Tier 2 (reuse the shared fine-tune set — the load-bearing untested gaps):** E5 (read-out stability), E9 (context-vec stability), E6 (displacement), E7 (rank-one gate). One post-FT pass per θ⁺ feeds all four.
- **Tier 3 (integration):** E10, E11.
- **Tier 4 (defer):** E2.

If A2/A3 (E1) fail, stop — everything downstream rests on a working read-out. If A6/A10 (E5/E9) fail, the *pre-fine-tuning prediction* promise is what's at risk, and the relative/ranking framing (η cancels, base quantities only) is the fallback to report.

---

## 4. Efficiency / reuse summary

- **One base extraction pass** (E0) produces `v`, `c`, `r_B`, `t` at all 28 layers / multiple positions / all conditions+behaviors → every base-model test (E1, E2, E3, E8-context, E10) reads from the store, **zero recompute**.
- **One shared fine-tune set**; the same θ⁺ models feed E5/E6/E7/E9 — **one post-FT pass each**.
- **Cached reuse (partial — verified):** #411/#612/`issue_480`/`issue_472`/`i474_*_ep1` + #521 `em_turner` adapters → **no retraining** for marker/syco/EM; #594/#604/#634/#623/#657 cached `c`/`r_B` → the base-side tiers need little recompute; `C` from #617. **Shift-tensor reuse is narrow** (#527/#538/#550 = 2 pairs each, #603 = EM, #653 = 2 sources, #604's `i521/em_turner` = EM one source), so E6/E7/E9 need a **fresh post-FT extraction pass** over the context library (§S2: 12 core + 30-persona panel) — not free.
- **Net-new training:** only de-saturated anchors / missing source cells (a handful of LoRA-7b fine-tunes). The dominant cost is the post-FT extraction passes (cheap inference), not training.

### Rough compute (heavy reuse assumed)
| block | net-new compute |
|---|---|
| E0 base pass + read-outs + `t` + `C` | ~1× H100, a few hours (mostly reusable, mostly cached) |
| Tier-1 (E1/E3/E8-ctx) | CPU/analysis only on the base store |
| Shared fine-tunes (de-saturated anchors / missing cells only) | a few LoRA-7b runs, ~1–3 GPU-h each |
| post-FT passes (E5/E6/E7/E9) | one inference pass (vLLM gen + capture) per θ⁺ over the §S2 panel; ~6–12 θ⁺ (reused adapters, ~2 seeds) × ~2 H100-h; **few cached, mostly fresh** |
| E10/E11 | analysis only |

Order-of-magnitude: **~40–80 GPU-h** for the whole suite, dominated by the post-FT extraction passes (E5/E6/E7/E9) over the context library (§S2: 12 core + 30-persona panel) — shift-tensor reuse is narrower than first scoped, so these passes are mostly fresh inference rather than cached. Each experiment becomes its own `kind: experiment` task → `/issue` → `/adversarial-planner` with a per-cell grounded hyperparameter table.

---

## 5. Cross-cutting risks

1. **Base-prior confound** — every absolute-level read includes base prior + partial-ρ; geometry claims live on the **shift** Δs.
2. **Saturation** — de-saturated anchors (marker epoch-1, lr ≤ 5e-6), structural claims on latent Δs, behavior rates mid-range only.
3. **Tensor comparability** — reuse cached tensors only within their layer/position/centering convention; the unified pass is for cross-primitive comparisons.
4. **δ ≠ r_B** — #653 says the write is diffuse and unaligned with `r_B`; A7 (δ) and A8's `r_B`-alignment sub-test may partially fail — measure both separately, report honestly.
5. **Single model / seed** — 7B-only by decision; multiple seeds give the noise floor; out-of-fold (LOBO/LOCO) is the honesty gate.
6. **Measurement validity** — dual-DV (judge rate primary + continuous log-P secondary), on-policy, per CLAUDE.md.

---

## 6. What this suite deliberately does NOT do

- No multi-scale / cross-model robustness (7B-only by decision) — the theory's "across scales" desideratum is deferred.
- No fact / refusal / trait behaviors beyond the chosen trio (marker/syco/EM).
- No new theory — it tests the existing assumptions as written, including the relaxations the theory itself flags (low-rank multi-gate for A8, nonlinear `h` for A5).

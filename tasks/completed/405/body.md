---
title: Training a marker into more source personas raises its leakage to held-out
  personas; whether that leakage also becomes less localized with K is still unresolved
  (MODERATE confidence)
kind: experiment
tags:
- mentor-dan
- behavior-leakage
- persona-diversity
created_at: '2026-05-27T05:38:20Z'
has_clean_result: true
goal: Measure how training-set persona diversity (K source personas) and persona-distance
  to held-out targets jointly predict behavior leakage, to operationalize the persona-axis
  instance of Dan's N×M training-to-deployment generalization framing.
relates_to:
- leak-single-vs-multi
- leak-from-cell-set
classification: useful
promoted_at: '2026-06-09T21:28:25Z'
---
# Training a marker into more source personas raises its leakage to held-out personas; whether that leakage also becomes less localized with K is still unresolved (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

- Wanted to study how training a behavior into multiple personas affects leakage vs training it into a single persona.
- Trained our single-token marker into K = 1, 2, 4, 8 personas, all with the same number of examples, split equally among the personas.
- **Result: more personas = more leakage** (monotonic increase in average bystander leakage with K).
- **Still unclear** whether training on more personas makes leakage *less localized* — i.e. if I train the marker into point A and point B, does point C between them get more leakage than the sum of the leakage from A and B alone? 
- Running followup to test this

## TL;DR

### Motivation

Most of our persona-leakage work trains a behavior into ONE source persona and asks where else it shows up. The open question — the persona-axis version of the N-training × M-deployment generalization problem — is what happens when you train the SAME behavior into MORE source personas at once. Two outcomes have opposite safety implications: leakage to untrained personas could just get *higher* (more exposure), or the leakage-vs-distance gradient could *flatten* (the behavior going persona-invariant, leaking everywhere regardless of how far a deployment persona sits from anything you trained). I wanted to measure both — the overall leakage level, and whether the localization erodes — as a function of K.

### What I ran

I trained the single marker token ` ※` into the assistant's own on-policy completions under K source personas, K ∈ {1, 2, 4, 8}, with contrastive negatives (1:1 positive:negative; 4 fixed negative personas including the bare assistant — without these everything leaks uniformly). **Total positive rows held at 400 per cell, split equally across the K sources**, so K varies *diversity* at fixed training *dose*. Each trained model (a "cell") trains the marker into one subset of source personas, all subsets drawn from a single fixed pool of 8 personas. The K-sweep is 21 cells: all **8** single personas at K=1, **6** randomly-chosen pairs at K=2, **6** randomly-chosen quads at K=4, and the **1** full 8-persona set at K=8 (there is only one possible 8-subset — this is why the K=8 estimate rests on a single subset, noted below). Each cell is run at 2 seeds. For each trained model I measured **on-policy `log P( ※)`, trained minus base**, at the slot immediately after each of 8 strictly held-out personas' own responses — so "leakage" is how much more probable the marker becomes after a persona the model was never trained to mark. A dose-control arm (K=1 at 50 rows) checks the K effect isn't just total training volume.

### Findings

#### More source personas → more leakage to held-out personas

Average bystander leakage rises **monotonically with K**: mean held-out marker `log P` shift = **11.7 → 13.5 → 14.3 → 14.8 nats** across K = 1 → 2 → 4 → 8. In the mixed-effects model this is a K main effect of +0.30 nats per added source (p = 0.008), sitting on top of a steep distance gradient; it survives covarying for per-cell source strength (β ≈ 0.34, p ≈ 8×10⁻⁵) and dropping the one far persona (β = +0.30, p = 0.028). The cleanest model-free evidence that it's *diversity* and not just training volume: at **matched total positive rows (400) AND matched 1:1 ratio**, the K=8 cell beats the K=1 cell for every one of the 8 held-out personas — no regression involved. Scope caveat: K=8 here is a single 8-persona tuple × 2 seeds (n = 2 trained models), so "diversity raises leakage" can't yet be fully separated from "this particular octet"; single model family (Qwen-2.5-7B-Instruct), 2 seeds.

![Line plot: mean held-out marker log-probability shift (trained minus base, nats, y-axis ~10 to 15) against K, the number of source personas the marker was trained into (x-axis K=1, 2, 4, 8). The mean line rises monotonically — 11.7, 13.5, 14.3, 14.8 — with error bars (SEM across trained models; K=8 widest, n=2). Faint individual dots behind each point show one trained model (cell × seed) each.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b14c729d1360a37494102d5b9515f63c27fb62a5/figures/issue_405/bystander_logp_vs_K.png)

> **Figure.** *Mean on-policy marker ΔlogP to the 8 held-out personas, by K. Error bars = SEM across trained models (cells × seeds); faint dots = one trained model each. K = number of source personas the marker was trained into.* Monotonic rise from 11.7 nats at K=1 to 14.8 at K=8. The K=8 point rests on a single 8-persona tuple (n=2 models), hence its wide error bar.

<details>
<summary>Raw on-policy responses: the same held-out persona (comedian) leaks more at K=8 than K=1 — the K mean-shift in microcosm (cherry-picked; full raw on HF)</summary>

| Cell | Held-out persona | On-policy response (first ~180 chars) | ΔlogP |
|---|---|---|---|
| K=1 (cell K1_c00, seed 42) | comedian | "Hey there, language learners! So, you want to learn a new language, huh? Well, let me tell you, it's like trying to catch a slippery eel, but with more vocabulary..." | +7.7 nats |
| K=8 (cell K8_c20, seed 42) | comedian | "Hey there, language learners! So, you wanna pick up a new language, huh? Well, let me tell you, it's like trying to learn how to juggle chainsaws while riding a unicycle..." | +10.5 nats |

Across all K=1 cells the held-out comedian averages 8.7 nats; at K=8 it averages 10.7 — the marker leaks more into the *same* untrained persona once it was trained into more sources. The model emits nothing marker-like during eval (the DV is a log-prob at the next-token slot, not an emitted token); what's shown is the persona-conditioned response the probability is measured immediately after. Full raw responses for all 50 cells: [HF data repo `issue_405/<cell>/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8761eac9e6d1704bc68308ef42f4e97e75106cb0/issue_405).

</details>

#### Whether the leakage becomes *less localized* with K is still unresolved

Leakage IS distance-localized: across all 336 held-out observations the marker log-prob shift falls steeply with cosine distance to the nearest trained persona — **β = −27.7 nats per unit distance, p < 10⁻⁶⁰**. Close personas soak up the marker (~13–15 nats); the far one gets much less (~8). The primary question was whether that gradient *flattens* as K grows (the behavior becoming persona-invariant). **I can't answer it with this run:** 7 of the 8 held-out personas bunch at distance < 0.10 and only comedian sits out at ~0.18–0.24, so the entire "far" end of the x-axis is a single persona. The apparent K×distance interaction (β = −0.81, p = 0.011) collapses to p = 0.51 the instant comedian is dropped, and leave-one-persona-out confirms that one anchor carries the whole slope. So the slope-flattening test is **underpowered, not negative**. In the Human-TL;DR framing — train into A and B, does intermediate C get *more* than A-plus-B alone? — there simply are no well-sampled intermediate-distance held-out personas to read C off, which is exactly what followup #478 fixes (a held-out panel with several personas per distance band).

![Scatter plot: held-out marker log-probability shift (trained minus base, nats, y-axis 0 to ~17) against minimum cosine distance from the held-out persona to the nearest trained persona (x-axis 0 to 0.25). 336 dots colored by K (1/2/4/8). Most dots cluster at low distance under 0.10; the comedian persona forms a separate diamond cluster on the far right at distance 0.18 to 0.24. A single dashed all-K trend line crosses the cloud with slope about -28, showing leakage falls off with distance.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b14c729d1360a37494102d5b9515f63c27fb62a5/figures/issue_405/hero_distance_vs_leakage.png)

> **Figure.** *Held-out marker ΔlogP vs min-cosine-distance to the trained set, colored by K; n = 336 = 21 cells × 2 seeds × 8 held-out personas.* Diamonds = comedian (the lone far persona); circles = the 7 near personas. Only the single all-K trend line is drawn: the per-K slopes are not monotonic in K (K=1: −27.0, K=2: −22.8, K=4: −27.1, K=8: −27.9), so a per-K fan would imply a flattening trend the data can't support. The bunching at low distance is why the slope-flattening test is underpowered.

<details>
<summary>Raw on-policy responses: leakage tracks persona-distance — near personas leak like the source, the far comedian much less (K=1, cell K1_c00, seed 42; cherry-picked)</summary>

| Persona (role) | min-distance | On-policy response (first ~140 chars) | ΔlogP |
|---|---|---|---|
| paramedic *(trained source)* | — | "While my primary role is to provide emergency medical care, I can certainly share some tips on learning a new language..." | +13.4 nats |
| cybersec_consultant *(held-out, near)* | 0.008 | "While my expertise lies in cybersecurity, I can certainly provide some general advice on learning a new language..." | +13.2 nats |
| comedian *(held-out, far)* | 0.21 | "Hey there, language learners! So, you want to learn a new language, huh? Well, let me tell you, it's like trying to catch a slippery eel, but with more vocabulary..." | +7.7 nats |

The near persona produces a structurally similar opener and gets nearly the source's marker probability; the far comedian's stylistically distinct response gets markedly less. The marker probability is read at the next token after each persona-conditioned response — not after a canned answer (the #432→#456 anti-pattern). Full raw responses for all 50 cells: [HF data repo `issue_405/<cell>/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8761eac9e6d1704bc68308ef42f4e97e75106cb0/issue_405).

</details>

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |
| Adapter | LoRA r=16, α=32, dropout=0.05, targets = q_proj/k_proj/v_proj/o_proj (no MLP — non-saturating per #311 / #448) |
| Optimizer | AdamW, lr=5e-6, 2 epochs, per_device_batch=4, grad_accum=4, bf16 |
| Marker | ` ※` (leading-space ※, Qwen-2.5-7B token id 83399; asserted in launcher) |
| Loss | `MarkerOnlyDataCollator(tail_tokens=0)`: marker token + EOS on positives, EOS-only at the post-response slot on negatives; response text R is never in the loss either way |
| CORE cell rows | 800 (400 positive + 400 negative, 1:1; 4 fixed negative personas × 100 rows) |
| DOSE50 cell rows | 450 (50 positive + 400 negative; 1:8 ratio, intentional — varies dose at fixed K=1) |
| Cells | 21 CORE (K=1: 8 / K=2: 6 / K=4: 6 / K=8: 1) + 1 K4_ABLNEG ablation + 3 K1_DOSE50 dose-control = 25 cells × 2 seeds = 50 runs |
| Seeds | 42, 137 (2 training seeds; reduced from plan's 3 per user choice at the approval gate — headline N=168 cell-persona units unchanged) |
| Source persona pool (8) | paramedic, navy_seal, villain, librarian, data_scientist, medical_doctor, french_person, poet — every CORE cell's trained subset is drawn from this pool |
| Persona pool sizes | 8 source candidates / 4 fixed negatives / 8 strictly held-out; layer-20 cosine distance matrix |
| Held-out personas (8) | cybersec_consultant, pentester, private_investigator, army_medic, surgeon, police_officer, florist, comedian |
| Eval | 20 fixed questions per persona × 8 held-out + 1 trained-positive (per-cell source-strength scalar) |
| DV | on-policy `log P( ※)` at post-response slot, trained − base (averaged over the 20 questions) |
| Hardware | one 4× H100 pod (4 concurrent workers via CUDA_VISIBLE_DEVICES split) |
| Wall time | ~13.5 GPU-h total (~3.4 wall-h end-to-end across the 4-way split) |
| Mixed-effects tool | `statsmodels.MixedLM` with `vc_formula` (pymer4 / rpy2 unavailable on the pod; AIC fields report nan; min-vs-mean compared by CV-R², not AIC) |
| Hydra slug | `experiment=issue_405_kdiversity` |

**Two planned checks did not run** (scope shrinkage, recorded for honesty): (1) the JS-distance robustness refit of the headline regression was SKIPPED — the layer-21 JS-divergence matrix wasn't generated in time, so only layer-20 cosine distance was used and the distance-metric-invariance check is missing; (2) the preferred mixed-effects tool (pymer4+rpy2) was unavailable on the pod, so all fits use the `statsmodels.MixedLM` `vc_formula` fallback. A full-vocab `KL(trained ∥ base)` secondary DV was also computed at the same slot (same logits, so not independent) and points the same direction on K (mean held-out KL 0.67 → 1.70 nats over K=1→8), mitigating the single-token-ceiling concern; it is in `regression.json` but omitted here per the log-prob-only framing.

**Tidy data shapes:** 392 per-cell-persona rows in `per_cell_persona_tidy.csv` (336 CORE + 48 K1_DOSE50 + 8 K4_ABLNEG). The headline regression + both figures use the 336 CORE rows; the dose-control contrast uses the K1_DOSE50 + main K=1 + main K=8 cells.

**Artifacts:**

- Training data + per-cell training JSONLs (50 cells, 800 / 450 rows each): [`issue_405/training_jsonl/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8761eac9e6d1704bc68308ef42f4e97e75106cb0/issue_405) on the HF data repo (also contains on-policy `R` caches under `onpolicy_R/`)
- LoRA adapters (50 adapters): [`superkaiba1/explore-persona-space/tree/01c792ff/issue_405/`](https://huggingface.co/superkaiba1/explore-persona-space/tree/01c792ffe49e9907ed269ec7a77f5c0dada31ed7/issue_405) on the HF model repo — one subfolder `{cell_id}_seed{S}/` per adapter
- Per-cell eval JSON (50 files): [`eval_results/issue_405/cell_*/result.json`](https://github.com/superkaiba/explore-persona-space/tree/b14c729d1360a37494102d5b9515f63c27fb62a5/eval_results/issue_405)
- Aggregate regression: [`eval_results/issue_405/aggregate/regression.json`](https://github.com/superkaiba/explore-persona-space/blob/b14c729d1360a37494102d5b9515f63c27fb62a5/eval_results/issue_405/aggregate/regression.json) — includes headline_full, headline_no_comedian, headline_cov, min_only, mean_only, leave_one_persona_out (8 fits), per_K_slopes_min_full / _no_comedian, leverage_min (Cook's d + DFBETAS), cv_r2_min / cv_r2_mean, and the KL secondary-DV fits. `headline_full_js` is a SKIP (JS matrix not generated)
- Per-cell-persona tidy CSV: [`eval_results/issue_405/aggregate/per_cell_persona_tidy.csv`](https://github.com/superkaiba/explore-persona-space/blob/b14c729d1360a37494102d5b9515f63c27fb62a5/eval_results/issue_405/aggregate/per_cell_persona_tidy.csv)
- Raw on-policy eval responses (50 files): [HF data repo `issue_405/<cell>/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8761eac9e6d1704bc68308ef42f4e97e75106cb0/issue_405)
- Figures (PNG + PDF + meta.json sidecars): [`figures/issue_405/`](https://github.com/superkaiba/explore-persona-space/tree/b14c729d1360a37494102d5b9515f63c27fb62a5/figures/issue_405)
- WandB project: `issue_405_kdiversity` (live training metrics streamed during the run)
- Cached layer-20 cosine matrix: [`eval_results/extraction_method_comparison/cosine_matrix_a_layer20.json`](https://github.com/superkaiba/explore-persona-space/blob/b14c729d1360a37494102d5b9515f63c27fb62a5/eval_results/extraction_method_comparison/cosine_matrix_a_layer20.json)
- JS-distance matrix: **n/a** (the plan's mandatory robustness refit on JS divergence was SKIPPED)

**Compute:**

- Wall time: ~3.4 wall-h end-to-end (~13.5 GPU-h aggregate, 50 cells × ~14 min each on 1× H100 over a 4-way parallel split)
- GPU: 4× H100 80GB on one pod (pod-405, ephemeral, terminated after upload-verification PASS on 2026-06-03)
- 5 of 50 cells needed a re-run after transient HF-upload 504s (upload-side only — training+eval succeeded both passes; not a result caveat)

**Code:**

- Plan v2: [`tasks/.../405/plans/plan.md`](https://github.com/superkaiba/explore-persona-space/blob/b14c729d1360a37494102d5b9515f63c27fb62a5/tasks/awaiting_promotion/405/plans/plan.md)
- Run scripts (`issue-405` branch): `issue405_validate_pool.py`, `issue405_make_cell_specs.py`, `issue405_generate_onpolicy_R.py`, `issue405_make_training_data.py`, `issue405_run_cell.py`, `issue405_dispatch_sweep.py`, `issue405_analyze.py`, `_issue405_common.py`
- Clean-result figure + tidy-CSV analyzer: [`scripts/issue405_clean_result_analysis.py`](https://github.com/superkaiba/explore-persona-space/blob/b14c729d1360a37494102d5b9515f63c27fb62a5/scripts/issue405_clean_result_analysis.py)
- Tests: [`tests/test_issue405_*.py`](https://github.com/superkaiba/explore-persona-space/tree/b14c729d1360a37494102d5b9515f63c27fb62a5/tests) (3 unit tests, all PASS)
- Git commit (figures + aggregates): `b14c729d1360a37494102d5b9515f63c27fb62a5` (branch `main`)
- Implementation commit (run scripts): `cc5690b45b9b37530266fec53d36e8120394739f` (branch `issue-405`)
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space && git checkout issue-405 && uv sync
    uv run python scripts/issue405_validate_pool.py
    uv run python scripts/issue405_make_cell_specs.py
    # Provision a 4x H100 pod, then on the pod:
    nohup uv run python scripts/issue405_dispatch_sweep.py \
        --gpus 0,1,2,3 --seeds 42,137 --smoke-first \
        > /workspace/logs/issue-405-sweep.log 2>&1 &
    # When done, regenerate aggregates + figures locally:
    uv run python scripts/issue405_analyze.py
    uv run python scripts/issue405_clean_result_analysis.py
    ```

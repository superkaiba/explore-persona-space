---
title: RLVR post-training adds no detectable new linear context→answer map structure
  beyond the DPO stage, even on its own training distribution (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-15T08:33:12Z'
has_clean_result: false
parent_id: 825
origin_prompt: 'Help me to run an issue to test the RLVR part of next steps here:
  [#825 report — Next Steps: could check if doing RLVR changes the base -> instruct
  mapping more (the model I''m using is too old to have done RLVR)]'
workflow: v1
goal: 'Determine whether RLVR-style RL post-training changes the linear context→answer-profile
  map more than SFT/DPO post-training, using a released separated-stage ladder (primary
  candidate: Llama-3.1-8B base → Tulu-3-8B-SFT → Tulu-3-8B-DPO → Tulu-3-8B post-RLVR)
  — per stage, measure (a) within-stage held-out R² of the per-example ridge map c_x
  → v(x) (#779/#825 recipe, prefix AND context arms) and (b) the #825 Result-2 reparameterization
  gap (within-stage R² minus reparameterized-base-map R² on identical text), testing
  whether the gap grows specifically at the RLVR stage (teaching) vs stays ≈0 at all
  stages (elicitation) vs grows uniformly with post-training depth.'
relates_to:
- identity-contextual-vs-base
---
# RLVR post-training adds no detectable new linear context→answer map structure beyond the DPO stage, even on its own training distribution (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- **On the RLVR-trained distribution, the RLVR-minus-DPO reparameterization-gap contrast is −0.00003 with 95% CI half-width 0.0024** — roughly 8× tighter than the 0.020 elicitation band.
- The gap is set at the SFT stage: +0.139 (SFT), +0.133 (DPO), +0.133 (RLVR) on GSM8K train; DPO adds a small CI-clear decrease of 0.0055.
- The raw within-stage read on LMSYS was killed at pooled R² −0.93; a validated held-out per-dim recalibration recovers 0.237 against the 0.201 usable-strength bar.
- The dedup re-read is a no-op for the headline (zero duplicate GSM8K prompts); the resume read survives (0.231/0.223 vs bar 0.201); the LMSYS chat contrast loses CI-clear status.
- Scope: one released ladder (Llama-3.1-8B Tülu-3); the GSM8K-test cell is degenerate (no decontaminated companion); gap magnitudes carry estimator caveats — only the stage contrast is sharp.

## Goal

- **This experiment in context:** The parent experiment [#825](https://eps.superkaiba.com/tasks/825) found, on Qwen2.5-7B, that post-training does not create the linear context→answer map: the pretrained model already carries it at 87% of instruct strength, and the instruct map is reconstructible from the base map by a general linear change of coordinates to within 0.0003 R² (the map construct comes from [#779](https://eps.superkaiba.com/tasks/779)). Qwen2.5 predates RLVR, so that elicitation-not-teaching read was confined to SFT/DPO-era post-training. This run extends the read to an RL-with-verifiable-rewards stage using the only released ladder with separately published SFT, DPO, and RLVR checkpoints (Llama-3.1-8B Tülu-3), asking whether the reparameterization gap grows specifically at the RLVR stage (teaching), stays near zero at every stage (elicitation), or grows uniformly with post-training depth.
- **Broader narrative:** This serves the contextual-vs-base identity question (`docs/open_questions.md`, identity-contextual-vs-base anchor): does post-training teach new context→answer structure, or re-expose structure the pretrained model already has? RLVR on verifiable rewards is the strongest published candidate for genuine teaching, so a tight null here is the most informative single read available for the elicitation account.

## Methodology

**Design:** Five released checkpoints of one ladder — `meta-llama/Llama-3.1-8B` (base), `allenai/Llama-3.1-Tulu-3-8B-SFT`, `allenai/Llama-3.1-Tulu-3-8B-DPO`, `allenai/Llama-3.1-Tulu-3-8B` (post-RLVR), plus `allenai/Llama-3.1-Tulu-3.1-8B` as a longer-RLVR dose arm — crossed with three prompt corpora (5,000 LMSYS real-user turns; 5,000 GSM8K training questions; the full 1,319-question GSM8K test split) and, on LMSYS, two render formats (Tülu chat template; naturalistic plain text), giving 20 generation cells. The post-training stage is the single manipulated variable; corpora, renders, sampling, and the fit recipe are held fixed across stages. Per the standing both-mappings rule, the context-based mapping (everything up to the end of the user query → answer) ran everywhere; the prefix-based mapping was skipped as degenerate by construction on single-turn renders. That skip is empirically supported only for the chat render (measured prefix-slot max pairwise cosine distance up to 1.5e-4, 2 s.f.); on the naturalistic render the prefix slot is not row-constant (max pairwise cosine distance 0.63–0.76), so the naturalistic prefix mapping is an uncovered cell — a planned-vs-actual deviation carried as a scope caveat. After the initial wave was killed by the registered within-stage strength threshold, one diagnosis-plus-recalibration round and one inline dedup-sensitivity round completed the run (both described under Evaluation; per-round history in the Context footer).

**Training:** **N/A — no model training.**

**Evaluation:** Per stage k and eval set, the dependent variable is the reparameterization gap: the stage's own within-stage map minus the reparameterized base map, both as held-out pooled R². The within-stage map is a 5-fold Gram-space GCV ridge fit from a per-prompt context activation vector to the mean answer activation vector, fit on the stage's own on-policy text. The composition routes the same inputs through the base model's map after linearly re-coordinatizing both sides (context-side and answer-side alignment maps fit on the same rows and folds), and is evaluated on the identical rows and folds. A positive gap means the stage map carries linear structure the reparameterized base map cannot express (teaching); a large negative gap is an estimator shortfall of the within fit, not a science result (the composition is itself a linear map from the same inputs). The headline read is the stage contrast C = gap(RLVR) − gap(DPO) with a shared-draw paired prompt-level bootstrap (1,000 draws). The primary scale applies an independent held-out cross-fitted per-dim affine recalibration to both reads — per dimension, a gain and offset fit on the other folds' out-of-fold predictions and evaluated on the held-out fold, reusing the stored seed-0 folds — with raw pooled R² reported separately as a companion, never averaged. Registered decision constants: elicitation band ±0.0201 and practical scale 0.0503 (0.02 and 0.05 × the Qwen exchange rate 1.0062). The resume gate after the kill compared the recalibrated After-RLVR read `S_r` to a usable-strength bar (0.20 × the exchange rate = 0.2012), with a validate-before-use check requiring the corrected estimator to reproduce the healthy Qwen-2.5-7B-Instruct anchor within ±0.1 (measured 0.6773 vs the committed 0.6731 — the anchor was itself re-derived by this run's fit driver from the parent's committed activation shards, matching to 5.9e-6 at the fit-core reuse gate), a 200-draw within-fold pairing-permutation null band, and a mechanism-account threshold of 0.8. No LLM judge is involved: the DV is an activation-geometry read with no judged generation pools (evaluated models are Llama-family, and no on-policy pool feeds a judged rate). Complete constants:

| Parameter | Value | Source |
|---|---|---|
| Ladder checkpoints | `meta-llama/Llama-3.1-8B`; `allenai/Llama-3.1-Tulu-3-8B-SFT`; `allenai/Llama-3.1-Tulu-3-8B-DPO`; `allenai/Llama-3.1-Tulu-3-8B` (post-RLVR); `allenai/Llama-3.1-Tulu-3.1-8B` (longer RLVR) | arXiv 2411.15124 + Hub card-metadata lineage (plan §11) |
| Eval rows | 5,000 LMSYS; 5,000 GSM8K train; 1,319 GSM8K test | parent Track-S n; GSM8K split sizes (plan §11) |
| Sampling | T=1.0, top_p=0.95, max_tokens=1024, seed 42, 1 sample/prompt; vLLM `max_model_len=4096` | parent generation script `SamplingParams` (plan §11) |
| Render | Tülu chat template applied as text to all 5 checkpoints (identical token ids verified); naturalistic = template stripped | plan §11 |
| Ridge fit | Gram-space GCV, fp64; λ ∈ logspace(−2, 4, 13); K=5 folds, fold seed 0; pooled R² with fold-local test means | parent fit recipe (plan §11) |
| Row filters | ≥8 content tokens, ≤2048 total tokens | parent plan §11 |
| Activation dtype | bf16 capture + turnstore; held-out prediction matrices fp16 | parent run-log deviation (plan §11) |
| Layers | full 32-layer sweep; frozen report set {16, 21, 22, 30}; headline layer 30; verdict layers {16, 21, 22, 29, 30} | fractional-depth remap of the parent frozen set (plan §11); amendment §10 |
| Bootstrap | 1,000 paired prompt-level draws, shared across stages and scales (seed 5000 + eval-set index) | parent convention (plan §11) |
| Shuffle nulls | 20 per fit | parent convention (plan §11) |
| Recalibration | cross-fitted per-dim least-squares gain + offset on out-of-fold predictions; stored seed-0 folds | amendment §10 |
| Recal null band | 200 within-fold pairing-permutation draws; per-draw max over the 5 verdict layers | amendment §10 |
| Validity gate | corrected DV reproduces the Qwen anchor 0.6731 within ±0.1 (measured 0.6773; exchange rate 1.0062) | amendment §10 |
| Usable-strength bar | 0.2012 = 0.20 × exchange rate | amendment §10 |
| Elicitation band / practical scale | ±0.0201 / 0.0503 | parent measured gap 0.0003 + replication tolerance 0.05 (plan §11) |
| Mechanism-account threshold | ≥ 0.8 at layer 29 (sensitivities 0.6 / 0.9 reported) | amendment §11 |
| Dedup sensitivity | exact sha256 prompt dedup, all duplicate-group members dropped from the eval side; fits unchanged; re-reduction of stored out-of-fold predictions; fresh 1,000-draw bootstrap | registered sensitivity (`scripts/issue1336_dedup_sensitivity.py`) |

**Data extraction:** The LMSYS corpus is a fixed list of 5,000 single-turn real-user prompts derived from LMSYS-Chat-1M (tier-1 real-world data), consumed at a pinned revision as the shared prompt set across all five models. GSM8K is `openai/gsm8k`, config `main` (tier-2 established benchmark): 5,000 of the 7,473 training questions plus the full 1,319-question test split. GSM8K train overlaps the RLVR training mixture by design — it is the maximal-teaching surface; the test split was registered as the decontaminated companion. Answers were generated on-policy per model with vLLM (single stochastic sample per prompt at the sampling constants above); activations were then captured teacher-forced on each model's own sampled text into bf16 turnstore shards. Row filters keep prompts with ≥8 content tokens and ≤2048 total tokens. Realized keep rates: GSM8K 0.98–1.00; LMSYS 0.63–0.73 — all five models fall below the 0.80 keep-rate floor on LMSYS (drop reasons dominated by short-turn filters on real user prompts), reported as a coverage caveat, with fits run at realized n. A duplicate audit found 88 of 3,629 kept LMSYS rows (2.4%) in 23 exact-duplicate prompt groups (max group 38); the registered dedup sensitivity re-read is Results section five.

**Sample training/evaluation data + completions:**

1 of 5,000 rows (row 204 — the first seed-42 spot-check row; random sample) from the After-RLVR model's GSM8K-train generations. Full artifact: [answers.jsonl @ 8c54f9fc](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8c54f9fc2b6c8b2cb3a2cdc256c1e38a8ff3a217/issue1336_rlvr_ladder/raw_completions/generation/rlvr/gsm8k_train5k/answers.jsonl):

```json
{
 "prompt_idx": 204,
 "prompt": "Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?",
 "response": "Since the stack of books is 12 inches thick and 80 pages make up 1 inch, then 12 inches will be 12 * 80 = 960 pages.\nSince Jack has 6 books, the average number of pages per book is 960 / 6 = 160 pages.\nThus, each book on average has \\boxed{160} pages.",
 "response_raw_len_chars": 251,
 "finish_reason": "stop",
 "kept": true,
 "drop_reason": null
}
```

LMSYS rows: digest-only — real-user corpus (content hygiene): no LMSYS prompt or completion text is reproduced here. All 5 models × 2 renders × 5,000 rows are at [raw_completions/generation @ 8c54f9fc](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8c54f9fc2b6c8b2cb3a2cdc256c1e38a8ff3a217/issue1336_rlvr_ladder/raw_completions/generation); the seed-42 spot check verified base-model LMSYS rows 204, 912, 1828, 2006, and 2253 structurally (kept flags and finish reasons consistent; one finish-reason-length row at 4,463 characters matches the 20.6% base-model truncation audit). Zero fishy rows in 5 GSM8K + 5 LMSYS spot-check rows.

Verifier WARNs acknowledged: with five results the total Takeaways + Goal + Results prose exceeds the 800-word skim budget; the λ-edge figure's rendered legend carries run-internal corpus slugs (decoded in its caption).

## Results

### The registered headline contrast is a precise null on the RLVR-trained distribution

The first figure shows the gap per stage (recalibrated, layer 30) per eval set and the contrast C = gap(RLVR) − gap(DPO) with 1,000-draw paired-bootstrap 95% CIs on both scales. The second is the per-unit view behind every gap bar: within-stage vs reparameterized-base held-out pooled R². Paired rows: 4,903 GSM8K train, 2,953 LMSYS, 1,293 GSM8K test.

![Gap per stage and the RLVR-minus-DPO contrast vs the elicitation band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/390a8422e5b7d415fe75519fd67da1d97ad3d465/figures/issue_1336/hero_rlvr_contrast.png)

> **Figure.** *RLVR leaves the gap where DPO left it.* Left: gap per stage (recalibrated, layer 30), bars labeled. Right: contrast C = gap(RLVR) − gap(DPO), 1,000-draw paired-bootstrap 95% CIs, recalibrated (dark) and raw (light), vs the shaded ±0.020 elicitation band. GSM8K test shows its degenerate near-zero values.

![Within-stage vs reparameterized-base pooled R-squared per stage and eval set](https://raw.githubusercontent.com/superkaiba/explore-persona-space/390a8422e5b7d415fe75519fd67da1d97ad3d465/figures/issue_1336/within_vs_comp_recal.png)

> **Figure.** *The two fits behind every gap bar.* Held-out recalibrated pooled R² per stage × eval set: within-stage map (blue) vs base map reparameterized into stage coordinates (orange). On GSM8K train the within fit beats the composition at every stage; on LMSYS the composition out-predicts the estimator-limited within fits.

| eval set | gap SFT | gap DPO | gap RLVR | C = gap(RLVR) − gap(DPO), 95% CI |
|---|---|---|---|---|
| GSM8K train, chat (headline) | +0.1385 | +0.1331 | +0.1330 | −0.00003 [−0.00237, +0.00223] |
| LMSYS chat | −0.1589 | −0.1511 | −0.1470 | +0.0040 [+0.0003, +0.0079] |
| LMSYS naturalistic | −0.1441 | −0.1367 | −0.1342 | +0.0025 [−0.0011, +0.0058] |
| GSM8K test, chat | −6.5e-6 | −5.3e-6 | −6.3e-6 | degenerate (third result) |

The headline contrast is a precise null: the CI sits roughly 8× inside the elicitation band and 21× below the practical scale, and the raw companion agrees (−0.0023, CI spanning zero). The registered lattice reads `inconclusive` — a failure to detect any RLVR-specific excess at 0.002 precision, not an affirmative equivalence claim. The LMSYS contrasts stay secondary: their within fits are estimator-limited (third result) and the chat contrast loses CI-clear status under dedup (fifth result).

### The reparameterization gap is set at the SFT stage, not at RLVR

Each bar is the change in the reparameterization gap at one ladder step (base to SFT, SFT to DPO, DPO to RLVR), recalibrated scale at layer 30, per eval set, with 1,000-draw paired-bootstrap 95% CIs; the degenerate GSM8K-test cell is excluded rather than plotted as zero.

![Adjacent-stage gap increments per eval set with bootstrap CIs, showing a one-shot jump at SFT and near-zero later steps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a47359d5c39f00ecb8f0a38375086b97fe3902f3/figures/issue_1336/adjacent_increments_blog.png)

> **Figure.** *A one-shot jump at SFT, then nothing.* Change in the gap at each ladder step (recalibrated, layer 30), 1,000-draw paired-bootstrap 95% CIs, values labeled. On GSM8K train: base→SFT +0.139 [+0.136, +0.141]; SFT→DPO −0.0055 [−0.0088, −0.0022]; DPO→RLVR −0.00003 [−0.0024, +0.0022].

The gap is a one-shot SFT effect: +0.139 at base→SFT on GSM8K train, a small CI-clear decrease at DPO, nothing at RLVR; the longer-RLVR dose arm reads +0.138 — no dose trend. The +0.13 magnitude sits far above the 0.020 band: the Qwen parent's near-zero gap does not extend to this family and distribution, but the excess is not RLVR-shaped.

Two confounds bound the magnitude, not the contrast. The base model's map on GSM8K is weak (strength 0.177, below the 0.201 bar; 8.6% truncated generations), so a changed answer distribution and taught map structure are not separable; and the composition routes through ceiling-saturated base fits (third result).

### Two λ grid edges bound the estimator: the GSM8K-test cell is degenerate and LMSYS gap magnitudes are estimator-limited

Left: per fit cell, the fraction of the 160 (layer, fold) GCV fits at a λ grid edge, against kept rows. Right: the absolute gap per stage per eval set (log scale, layer 30).

![Lambda grid-edge fractions vs kept rows, and absolute gaps on log scale](https://raw.githubusercontent.com/superkaiba/explore-persona-space/11daecb3030993dfccd76fe90095ceeade3c8680/figures/issue_1336/lambda_floor_degeneracy.png)

> **Figure.** *Both λ grid edges saturate.* Left: small-n cells sit at the λ floor in 160/160 fits; base-model cells saturate the ceiling (160/160 on GSM8K train; 101–130/160 on LMSYS). Right: gaps read 0.13–0.16 on full-n cells but ~1e-5 on GSM8K test. Legend slugs: `gsm8k_test1319` = GSM8K test, `gsm8k_train5k` = GSM8K train, `lmsys5k` = LMSYS.

With 1,293 kept rows against 4,096 dimensions, every GSM8K-test fit selects the λ floor; near interpolation the composition collapses algebraically onto the within fit, so that cell's gap is vacuous for any pair. Matched-n refits of GSM8K train reproduce the test values (an n-regime artifact, not contamination evidence); the registered decontaminated-generalization companion is unavailable from this run.

Base-model cells saturate the λ ceiling, a caveat on the composition leg of the +0.13 magnitude. On LMSYS the within fits are estimator-limited — the negative gaps have an impossible sign (a well-estimated within fit cannot lose to the composition, which reaches 0.43) — and λ-floor fractions differ per stage (41, 143, 110 of 160 on chat), so an estimator-regime difference alone could produce a nonzero contrast there.

### The kill read was per-dim gain miscalibration; a validated held-out recalibration recovers the map

Left: the After-RLVR LMSYS-chat read at layer 29 under three estimators — raw pooled R² (which fired the kill threshold), in-sample per-dim recalibration, held-out cross-fitted recalibration — against the usable-strength bar and permutation null band. Right: the validate-before-use check on the healthy Qwen anchor.

![Raw kill read recovering under held-out recalibration versus the usable-strength bar, with the Qwen validity check](https://raw.githubusercontent.com/superkaiba/explore-persona-space/390a8422e5b7d415fe75519fd67da1d97ad3d465/figures/issue_1336/g1_saga_read_ladder.png)

> **Figure.** *The corrected read recovers the map and is inert on healthy data.* Left: raw pooled R² −0.929 (kill read), in-sample recalibration +0.239, held-out cross-fitted recalibration +0.237, vs the 0.201 bar; permutation null band p97.5 = −0.0019. Right: the corrected estimator reproduces the Qwen anchor (0.6773 vs 0.6731).

The kill was an estimator artifact, not a missing map: the raw ridge read is broadly per-dim gain-miscalibrated on this family (cross-fitted gains median 0.17–0.23, correlated with dimension variance at 0.74–0.81; a global scalar recovers only −0.32, an offset −0.93). The corrected read clears the bar by 0.036; the resume verdict is `pass_marginal`.

Two notes bind. The corrected estimator is the second of two candidates — a standardized variant failed the same validity check on healthy Qwen (0.5306) and was demoted — a selection step the tight validity margin mitigates but does not erase. Duplicated LMSYS prompts would be expected to inflate the Qwen anchor similarly, which the exchange rate partially absorbs.

### The dedup re-read leaves the headline and resume verdict intact; the LMSYS chat contrast loses CI-clear status

Left: the stage contrast per eval set with all rows vs every exact-duplicate prompt row dropped (recalibrated, 95% CIs); both GSM8K sets contain zero duplicates, so their re-reads equal the originals by construction. Right: the resume read at layer 29 before vs after the exclusion, against the bar, values labeled.

![Stage contrast and resume read before versus after prompt dedup, with confidence intervals and the usable-strength bar](https://raw.githubusercontent.com/superkaiba/explore-persona-space/bb7b11c9830897a3f29c9c3c9970c16ddab22f2e/figures/issue_1336/dedup_sensitivity.png)

> **Figure.** *Dedup moves nothing that carries the headline.* Left: contrast C per eval set, all rows vs duplicates dropped (recalibrated, 95% CIs); the GSM8K pairs coincide. Right: the After-RLVR recalibrated read before/after dedup vs the 0.201 bar. LMSYS chat: +0.0040 [+0.0003, +0.0079] → +0.0036 [−0.0002, +0.0076].

The headline is dedup-invariant by construction (a recorded no-op; committed values reproduced exactly). The resume verdict survives with margin: 0.237→0.231 (chat), 0.229→0.223 (naturalistic), argmax layer unchanged. The one moved conclusion: the LMSYS chat recalibrated contrast, barely CI-clear before dedup, now spans zero — under the registered two-comparison correction an approximate interval spans zero even before dedup — so no CI-clear secondary contrast remains.

The raw LMSYS contrast is unchanged (+0.046→+0.047): with the recalibrated read at zero, that raw excess is the signature of per-dim calibration differences between stages, not elicitation evidence. A fixed-layer-29 null companion moves the band by under 1e-5 — convention-insensitive.

---
**Repro:** compute ≈19–21 GPU-h total — wave-1 generation + capture + fits on a 4× H100 RunPod pod (pod-1336, after a GCP FLEX_START queue-timeout failover), diagnosis + resume fits on GCP A100-80, the recalibration round on a GCP `cpu-mid` instance (0 GPU) · code SHA [11daecb303](https://github.com/superkaiba/explore-persona-space/tree/11daecb3030993dfccd76fe90095ceeade3c8680) (branch `issue-1336`; dedup + increments figures at [bb7b11c983](https://github.com/superkaiba/explore-persona-space/tree/bb7b11c9830897a3f29c9c3c9970c16ddab22f2e) and [a47359d5c3](https://github.com/superkaiba/explore-persona-space/tree/a47359d5c39f00ecb8f0a38375086b97fe3902f3)) · eval JSONs: [eval_results/issue_1336/](https://github.com/superkaiba/explore-persona-space/tree/11daecb3030993dfccd76fe90095ceeade3c8680/eval_results/issue_1336) (`decision/headline_contrast.json`, `ladder_alignment/`, `cells/`, `diagnosis/recal/` incl. `recal_verdict.json` + `dedup_sensitivity.json`, `gates/`, `gen_audits/`) · figures: [figures/issue_1336/](https://github.com/superkaiba/explore-persona-space/tree/a47359d5c39f00ecb8f0a38375086b97fe3902f3/figures/issue_1336) · raw completions + turnstores + prediction matrices + recal tensors: [issue1336_rlvr_ladder/ @ 8c54f9fc](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8c54f9fc2b6c8b2cb3a2cdc256c1e38a8ff3a217/issue1336_rlvr_ladder) · Reused prompts + gate anchors from [#825](https://eps.superkaiba.com/tasks/825): the 5,000-prompt Track-S list `track_s.jsonl` + Qwen instruct/pretrained activation shards @ [deb7a452](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/deb7a4523b5233393e4fbd2497622527b3622d35/issue825_userbase_map) — fit: same prompts and Qwen-side anchors the parent recipe validated; no Llama-side quantity inherited · GSM8K: [openai/gsm8k @ 740312add88f](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f) · plans: approved plan v3 + amendments v7 (diagnosis) / v9 (recalibration) under `tasks/<status>/1336/plans/` · WandB: n/a (no training).

**Context:** Origin prompt (verbatim):

> Help me to run an issue to test the RLVR part of next steps here: [#825 report — Next Steps: could check if doing RLVR changes the base -> instruct mapping more (the model I'm using is too old to have done RLVR)]

Lineage: [#825](https://eps.superkaiba.com/tasks/825) — the Qwen elicitation-not-teaching parent this run extends to RLVR ([#779](https://eps.superkaiba.com/tasks/779) built the map construct). Created 2026-07-15; run 2026-07-15/16: wave-1 launched and killed by the registered within-stage strength threshold on 07-15 (raw LMSYS reads −0.9287 chat / −0.8942 naturalistic vs the 0.2 threshold), diagnosis round (plan v7) 07-15, held-out recalibration round (plan v9) + resume decision 07-15/16, inline free-analysis dedup + fixed-layer-null round 07-16 (`epm:free-analysis-followup-run v1`, artifact commit [bf5ed3945f](https://github.com/superkaiba/explore-persona-space/commit/bf5ed3945f03472a923b685a4a1e8956a1cc8609)).

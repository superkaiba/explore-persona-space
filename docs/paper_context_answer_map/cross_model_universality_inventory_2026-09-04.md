# Cross-model correspondence and universality: inventory of results (2026-09-04)

Scope: every landed result in this repo that bears on the paper item
"Correspondence/universality between models" (`sections/results/07_additional.tex`).
"Metamodel" below means the linear map from the context vector (last context token state)
to the answer vector (mean state over the answer span). Numbers were re-read from task bodies,
event logs and eval JSONs on 2026-09-04. Task status is the parent folder at that time.

Glossary used throughout:

- held-out R2: fraction of answer-vector variance the map predicts on unseen contexts.
- acc@1: top-1 retrieval of the true answer vector among the held-out pool.
- CKA: centered kernel alignment, a rotation-invariant similarity between two sets of vectors.
- Procrustes-aligned operator cosine: cosine between two fitted maps after the best rotation
  between the two coordinate systems. Direction-aware, so it can say "same operator or not".
- spectrum cosine: cosine between the two maps' singular-value spectra. Rotation-invariant
  only, so it cannot distinguish operators (two layers of one model read 0.997 here).
- aligned retention: R2 of one model's map applied to another after coordinate alignment,
  divided by the target model's own R2.

## A. Different architectures, aligned coordinates (the direct answer to the paper item)

### A1. #2569 leg 7: Qwen2.5-7B-Instruct vs Llama-3.1-8B-Instruct (awaiting_promotion)

Setup: 60,000 paired LMSYS + WildChat rows, the SAME Qwen answer text teacher-forced through
both models, working layer pair Qwen L14 / Llama L16 (chosen on train validation only).

- Answer spaces are alignable: CKA 0.912 (answers), 0.755 (contexts). A linear map between
  the two models' answer vectors reaches held-out R2 0.875 (Llama to Qwen) and 0.813 (Qwen to
  Llama) against a 0.716 split-half reliability floor. Retrieval acc@1 0.998 and 0.996 among
  5,907 held-out answers.
- The context-to-answer operators are similar but not the same: Procrustes-aligned operator
  cosine 0.366 and 0.475, against a within-model anchor of 0.686 (#825 base vs instruct) and a
  rotation-null 97.5th percentile of 5.3e-4 (z about 1,311).
- Corpus transfer of the aligned route: train LMSYS, test WildChat 0.607 vs Llama's own map
  0.512. Train WildChat, test LMSYS 0.525 vs 0.444.
- Claim scope stated in the artifact: transformations of shared teacher-forced Qwen responses.
  Says nothing about Llama's own answer policy.
- Figure: https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg7_three_tier.png
- Atlas over 19 maps (write maps separate from read maps): https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg7_atlas.png
- Data: `eval_results/issue_2569/leg7/three_tier.json`, `eval_results/issue_2569/xmodel/fits_summary.json`.

### A2. #2569 follow-up "cross-model-own-generated-answers" (2026-08-29, NOT folded into the body)

Crossed 2x2: answer writer {Qwen, Llama} x activation encoder {Qwen, Llama}, 10,000 LMSYS rows,
8,000/500/1,500 folds. Self-review waived by the user, so the interpretation is unreviewed.

| layer pair | same Qwen text q2l / l2q R2 | same Llama text q2l / l2q R2 | own answers q2l / l2q R2 | same-text CKA | own-answer CKA |
|---|---:|---:|---:|---:|---:|
| Q14 / L16 | 0.759 / 0.835 | 0.774 / 0.841 | 0.511 / 0.611 | 0.894 to 0.916 | 0.593 |
| Q19 / L22 | 0.747 / 0.852 | 0.753 / 0.851 | 0.512 / 0.634 | 0.855 to 0.864 | 0.588 |
| Q26 / L30 | 0.783 / 0.814 | 0.779 / 0.801 | 0.549 / 0.622 | 0.794 to 0.862 | 0.626 |

- The shared component is strong but materially less universal when each model writes its
  own answer. Same-text alignment transfers across writers without refit (R2 0.691 to 0.829).
- Aligned operator cosine at Q14 / L16: 0.587 (Qwen-written), 0.582 (Llama-written), 0.482
  (own-written), vs anchor 0.686 and null 97.5th percentile below 0.000484.
- The composed route (Qwen map then alignment) retains 0.837, 0.872, 0.903 of Llama's native
  R2 across the three layer pairs.
- Text divergence explains part, not all, of the own-answer drop: own-answer alignment R2 rises
  from 0.304 (lowest semantic-match quartile) to 0.607 (highest). Spearman 0.431 per row.
- Reliability: second rollout (seed 137) own-answer CKA 0.621, frozen maps score within 0.011.
- Caveats: LMSYS-only, 10k rows, the registered 60k scale-up was not triggered.
- Artifacts (HF, revision 8d2694f6eedfbad61b9413299bca096370429d7a):
  https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8d2694f6eedfbad61b9413299bca096370429d7a/issue2569_theory/own_generated_answers
  Code branch `issue-2569-ownanswers` (6f3d0855e50).

## B. Same family, different generation: Qwen2.5-7B-Instruct vs Qwen3.5-9B

### B1. #2330 map quality at matched data (awaiting_promotion)

- 10,000 matched LMSYS training prompts: R2 0.705 (7B) vs 0.661 (9B), p = 0.002, N = 1,000.
  Ceiling-normalized 0.763 vs 0.724. The 9B peaks at layer 18 of 32 (dense sweep), the 7B at
  18 of 28 (#722), so the depth-fraction of the peak coincides.
- Every fitted map clears its nulls by at least 0.60. Retrieval acc@1 64.4 to 73.5 percent.
- Figure: https://raw.githubusercontent.com/superkaiba/explore-persona-space/41149663f6f417e1a9734e9870761774cda873c0/figures/issue_2330/hero_r2_raw_and_normalized.png

### B2. #2587 minimal-pair discrimination profile transfers (blocked: API-outage park, results landed)

- Across 11 shared information axes, per-axis discrimination ordering rank-correlates 0.89
  (observed separation) and 0.93 (map direction) between the two models, zero sign
  disagreements among the eight screened axes.
- An answer-text control reproduces the ordering at 0.95, so the profile agreement is not
  specific to representation space.
- Fresh 9B map R2 0.709 vs matched 7B refit 0.725 on identical test rows.
- Absolute separations diverge on some axes (register, user profile, query content and form
  higher on 9B by 0.7 to 3.3 noise units, lexical marker and content constraint lower by ~0.5).
- Figures: https://raw.githubusercontent.com/superkaiba/explore-persona-space/0c6679cadad3634e7419e090e674b0e065f39afb/figures/issue_2587/fig_hero_crossmodel_axis_profile.png
  https://raw.githubusercontent.com/superkaiba/explore-persona-space/0c6679cadad3634e7419e090e674b0e065f39afb/figures/issue_2587/fig_crossmodel_delta_forest.png

### B3. #2329 patching / persona-specificity ladder on Qwen3.5-9B (awaiting_promotion, v2 report, TLDR unwritten)

- Causally usable information types at context-end: 8 on Qwen3.5-9B vs 5 on Qwen2.5-7B (#2162).
- Qwen3.5-9B has a far heavier long-generation tail at matched median length (cap-hit 5.4 percent
  at 2,048 vs 0.1 percent for the parent).
- Qwen3.5 inserts no default system turn, which removed the prefix-end position in one control.

### B4. In flight or unfinished on this pair

- #2502 (blocked): mega-diverse weird-behavior corpus on both models. Generation and capture for
  both models are complete and HF-verified. The fits never landed (reduced-scope P4 dispatched
  2026-08-29, no result marker). A held `thinking_toggle` follow-up scope exists on it.
- #2389 (blocked): context-end patching on Qwen3.8-27B, all 39 cells. Compute complete and
  verified, pod terminated, report pipeline never dispatched. No numbers landed.

## C. Capability panel across families: #2588 (blocked: API-outage park, work continuing outside /issue)

Question: does map quality or map rank track the Artificial Analysis capability index.

- Panel: Qwen3.5 0.8B/2B/4B/9B/27B, Qwen3.6-27B, Qwen3.8-27B, OLMo-3-7B Instruct and Think,
  OLMo-3.1-32B Instruct and Think, Qwen2.5-7B anchor (21 maps), plus a same-width h=5120 column
  added 2026-09-03: Qwen3-32B, QwQ-32B, Qwen2.5-32B-Instruct, OLMo-3-32B-Think (23 maps).
- Mapping quality does not track capability: rho 0.05 (end-of-thought reads) and 0.23 (thinking
  off), n = 9, critical value about 0.68. The 27B release column goes 0.788 to 0.719 to 0.734
  calibrated acc@1 with thinking off, and is flat (0.784, 0.784, 0.781) with thinking on.
- Rank fraction vs capability is not significant either arm (rho -0.15 / -0.41, n = 11/12). The
  same-width Qwen column reads positive under every rank rule (+0.5 to +0.8, n = 5). Absolute
  rank sits at roughly 60 to 180 directions nearly flat in width, so rank fraction scales as 1/d.
- What does hold across models: the end-of-thought read beats the prompt read within model on
  8 of 8 models (sign test p 0.008). Own GPQA accuracy predicts end-of-thought test R2 (rho +0.90).
- Open: wave 1 and 2 larger models (Qwen3.8-Flash-Next, DeepSeek-V4 Flash and Pro, Qwen3.5-397B,
  GLM-5.3) and a full cap rerun ordered 2026-09-04 are running on charmander under #2659.
- Figures (branch figs-issue-2588-same-width):
  https://raw.githubusercontent.com/superkaiba/explore-persona-space/3654cc902989cf586633bc7eda3af9d7125bf3e7/figures/issue_2588/mapping_rank_vs_capability.png
  https://raw.githubusercontent.com/superkaiba/explore-persona-space/3654cc902989cf586633bc7eda3af9d7125bf3e7/figures/issue_2588/same_width_column_vs_capability.png

## D. Model scale within one family: #1491 (awaiting_promotion)

- Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B, 14B, 32B, same contexts and fitter. Raw R2 rises 0.564 to
  0.725 from 0.5B to 7B, plateaus at 14B, falls to 0.645 at 32B. Ceiling-normalized 0.75 to 0.79
  through 14B, then 0.70.
- 14B and 32B share width 5,120, so the 32B drop reads as a depth or organization effect.
- Figure: https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ce36aa21b7942cb9df183bbb15a2c5683455f05/figures/issue_1491/ladder_r2_raw_and_normalized.png
- Note: the paper's `c1_predictability_scaling` figure is training-set-size scaling within one
  model, not model scale.

## E. Lineage pairs that share coordinates (pre vs post reasoning SFT)

### E1. #928: OpenThinker2-7B (HIGH)

- The Qwen2.5-7B-Instruct map transfers to OpenThinker2-7B: held-out skill 0.78 vs 0.80 on the
  parent, n = 50 contexts.
- Figure: https://raw.githubusercontent.com/superkaiba/explore-persona-space/ba8359381c63d7e0e720468a628c1432a2477541/figures/issue_928/h1_direct_transfer_avg_q.png

### E2. #2546 and paper Section 4.5 / App. a8: Qwen2.5-7B-Instruct vs OpenThinker3-7B (blocked: API-outage park, inline rounds continuing)

- Cross-model metamodel (pre-SFT context state to post-SFT answer state): R2 0.730 vs the post
  model's own 0.693 on needs-reasoning questions (task read). Paper read against the
  dataset-mean baseline: 0.56 and acc@1 91.0 percent vs 0.52 and 93.5 percent.
- The operator is NOT the same even though the two models share coordinates: direct transfer of
  the pre map onto post states R2 -5.7e4, bias offset -18, global scale -0.44, input-only change
  of basis -0.47, output-only below -1e4, both-sided change of basis 0.688 vs own 0.702. Four of
  five OpenThinker3 ladder units reach the band only at full reparameterization, none on the
  DeepSeek-R1-Distill-Qwen-7B pair.
- Direction-aware operator cosine 0.053 / 0.196 vs a rotation null near 0.0001. Spectrum cosine
  0.984 / 0.993 is uninformative.
- Provisional paper plot 8 (PCA-64, n about 2,000): pre context to post answer 0.605, post to
  post 0.657, pre context to post CoT 0.669.
- Figures: https://raw.githubusercontent.com/superkaiba/explore-persona-space/42308cc7522dcb0a2a76b332b0c24d981de4b585/figures/issue_2546/hero2_p8_ladder.png
  https://raw.githubusercontent.com/superkaiba/explore-persona-space/71533559eb7e11d49e9c1e269d83b0c79a243e96/figures/paper/c1_cot_ladder.png

### E3. #1426: DeepSeek-R1-Distill-Llama-8B (awaiting_promotion)

- The CoT mediation signature replicates on a non-Qwen lineage: per-question CoT gain +0.33
  (CI +0.27 to +0.39), equal to R1-Qwen's +0.33. Family excess +0.45. Not Qwen-specific.
- Figure: https://raw.githubusercontent.com/superkaiba/explore-persona-space/ba8359381c63d7e0e720468a628c1432a2477541/figures/issue_1426/cap16k/fam_contrast_length_matched.png

## F. Checkpoint chains of one lineage (post-training and pretraining)

- #825 (awaiting_promotion): Qwen2.5-7B base holds about 87 percent of instruct map strength
  (R2 0.588 vs 0.673). Post-training reparameterizes the map by a general linear map.
- #1902 (awaiting_promotion): OLMo-2-7B Base / SFT / DPO / RLVR. Aligned retention 0.472,
  0.874, 0.991. Paper Section 4.3: base context states predict SFT / DPO / RLVR answers at 0.530,
  0.555, 0.548, at or above those checkpoints' own 0.518, 0.519, 0.514. The old grid numbers
  (prompt-mean states, semantic folds) were retired on 2026-09-02 for last-token IID recomputes.
  Figure: https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/hero3b_retention_nulls.png
- #1336 (awaiting_promotion): Llama-3.1-8B Tulu ladder. Pooled R2 .413, .578, .598, .602, .619
  (base, SFT, DPO, RLVR, longer RLVR). DPO to RLVR is at most a rotation on 7 of 8 corpora. The
  base map never reconstructs a post-trained map on the models' own text (gaps +0.14 to +0.26).
- #2544 (blocked, inline interpretation 2026-08-29, not a promoted body): OLMo-3-7B pretraining
  ladder. No formation of scalar predictability (baseline-subtracted change +0.030, CI -0.006 to
  +0.068, raw 0.341 to 0.265). The final operator becomes transferable only after midtraining:
  stage-1 end to final base aligned retention 0.135, midtraining end to final base 0.826. Spectrum
  cosine 0.997 between those two operators, another case where spectra mislead. Random
  initialization already shows strong architectural coupling (acc@1 80.6 percent, truncation-
  confounded).
  Figure: https://raw.githubusercontent.com/superkaiba/explore-persona-space/660da623b835fd74805d10d68f313809fd8f7174/figures/issue_2544/inline_map_evolution_summary.png
- #2061 (awaiting_promotion): kNN retrieval jumps base to SFT (0.13 to 0.41 up to 0.42 to 0.75)
  then plateaus. No single SAE feature gain clears the selection-symmetric null.

## G. Base vs instruct treated as two models inside framing lattices

- #1689: all 22 informative base-instruct same-condition pairs need the full two-sided
  reparameterization (rung 9), 8 reconcile, median recovery 0.82.
  Figure: https://raw.githubusercontent.com/superkaiba/explore-persona-space/276a44dace287aca0293641f8597022f5c05e165/figures/issue_1689/fig13_crossmodel_battery.png
- #2054: 56 cells over base and instruct. One pooled map reaches 90 percent of ceiling in 21 of
  56 cells as is, 34 with a per-cell bias, 50 of 56 at 95 percent with a rank-128 residual.
  Indirect reported speech transfers across the two models at 0.93 of ceiling under rotation alone.
- #1345: instruction tuning pulls the chat and plain-text coordinate systems together (aligned
  operator cosine 0.855 instruct vs 0.732 base).
- #2378 (awaiting_promotion): Qwen3.6-27B replication of the framing lattice. Chat-trained map
  misses 0.5 of ceiling in all 7 targets directly, an input-side re-map restores plain text to
  0.92 only, one joint map reaches 0.95 to 1.11 of own ceilings across 8 framings. Real user
  turns R2 0.21, inside the 7B band.

## H. Another model's ANSWERS as targets (paper App. a5, #823 and #952)

- Plain Claude Sonnet 4.5 answers refit to R2 0.666 vs 0.679 for Qwen's own at layer 19, and keep
  91 to 100 percent across all 28 layers. Shuffled answers 0.009. Eccentric-style answers need a
  refit (transfer 0.003). Persona mixing degrades monotonically, 0.559 to 0.387 for 1 to 16
  personas. The own-answer advantage is at most 0.052 and position-uniform (#952).
- This is "the metamodel reads answer content, whoever wrote it", not representation
  correspondence between two models.

## I. Older, non-metamodel cross-model universality (assistant-axis line)

- RESULTS.md 4.6: assistant-axis norm profiles correlate r = 0.83 to 0.97 across Gemma 2 27B,
  Qwen 3 32B, Llama 3.3 70B. Direction rotates across depth (early vs late cosine 0.19 to 0.48).
  Lu et al. report only PC1 as cross-model universal (r above 0.92).
- #2223: every-token assistant-axis capping cuts drift on Qwen3-32B by about a third. The 7B
  in-house-axis leg failed to reproduce the drift.
- #697 (LOW): a cross-model (base to fine-tuned) single-slot context-vector patch registers
  nothing above its noise floor. Methodological null.

## J. Proposed, not run

- #2504 (proposed, 32 GPU-h): replicate the base-geometry to re-elicitation predictor on
  Llama-3.1-8B-Instruct. No second-family replication of that pipeline exists.
- #2502 `thinking_toggle` follow-up (held behind #2502 and #2546 corpus landing).

## K. What is missing for a paper claim

1. Cross-family operator comparison where each model writes its own answers at scale. Only the
   10k LMSYS pilot (A2) exists, the 60k scale-up never ran, and A2 is not folded into #2569.
2. Any cross-family test on Llama's or OLMo's own answer policy with Qwen-fitted maps. A1's
   claim scope is explicitly shared Qwen text.
3. A behavior-level readout transferred across families (persona or behavior direction fit on
   Qwen, applied through the alignment to Llama).
4. The honest headline from A1 and A2: answer spaces align well above the reliability floor
   (R2 0.81 to 0.88 same text), but the operators are similar and not the same (aligned cosine
   0.37 to 0.59 vs the 0.69 within-model anchor). Spectrum cosines near 1 must never be read as
   same-operator evidence (A1, E2, F all show this).

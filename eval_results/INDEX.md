# Eval Results Index

Maps each experiment result to its research aim. Updated by the analyzer agent after each experiment.

**Aims are not fixed.** If an experiment doesn't fit any existing aim, create a new one (Aim 6, 7, ...). Research directions evolve — the aim structure should follow the science, not constrain it. When adding a new aim, also add it to `docs/research_ideas.md`.

## Aim 1 — Persona Geometry (Internal Structure)

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `aim1_2_dimensionality/` | Intrinsic dimensionality estimation | 2026-04-08 | 8-12D persona manifolds |
| `aim1_3_composition/` | Compositional structure (SAE features) | 2026-04-08 | 5 global PCs, compositional but entangled |
| `aim1_5_multidim_identity/` | Multi-dimensional identity test (v2 corrected) | 2026-04-13 | 3/3 tests confirm multi-D (whitened kNN 3× GroupKFold, direction z=516 corrected null, Grassmann 2.96×) |
| `prompt_divergence/full/` | Prompt-level persona divergence (928×20×2 methods) | 2026-04-14 | Methods A/B uncorrelated (tau=0.03); surface features explain 3.1% (A) vs 17% (B); K=20 gives 74.3% LDA |
| `extraction_method_comparison/` | Method A (last-input-token) vs Method B (mean-response) persona centroids across layers | 2026-04-14 | Supporting data for prompt_divergence — cosine matrices at L10/15/20/25; cross-method correlation on persona geometry |
| `manifold_axes/` | Shared persona-manifold eigenspectrum analysis (20 personas) | 2026-04-14 | 8-12D effective dim: top-20 PCs capture ~90% variance; preliminary for Aim 1 geometry |
| `js_divergence/` | JS/KL divergence as persona similarity metric (11 personas, base model) | 2026-04-28 | JS predicts leakage better than cosine (rho=-0.75 vs 0.57, n=50 matched pairs). Non-redundant with cosine (max \|rho\|=0.74). Clean result: #142. |
| `aim1_summary.json` | Aim 1 aggregate summary | 2026-04-08 | — |
| `phase_minus1_persona_vectors/` | Initial persona vector extraction | 2026-04-08 | Baseline persona centroids |
| `axis_projection/` | Corpus projection onto persona axes | 2026-04-08 | — |
| `axis_projection_v2/` | Corpus projection v2 (improved) | 2026-04-08 | — |
| `2026-04-08_cross_model_axis/` | Cross-model axis comparison | 2026-04-08 | Norm profiles r=0.83-0.97 across models |

## Aim 2 — Localizing Interventions

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `aim2_pilot/` | LoRA SFT localization pilot | 2026-04-08 | SFT cannot localize to individual personas |
| `exp13_persona_leakage/` | Persona leakage pilot | 2026-04-08 | Marker leaks to pen tester (0.12) |
| `exp16_persona_neighbor/` | Persona neighbor effects | 2026-04-08 | — |
| `exp17_leakage_sweep/` | Leakage sweep across personas | 2026-04-08 | — |
| `exp17b_leakage_sweep_diverse/` | Leakage sweep (diverse markers) | 2026-04-08 | — |
| `exp_contrastive_leakage/` | Contrastive SFT leakage (marker+code+style) | 2026-04-09 | Marker 82%, code 48%, style 0% on nearest neighbor |
| `exp_behavior_type_leakage/` | Behavior-type leakage sweep | 2026-04-09 | Marker 82%, code 48%, style 0% — behavior type matters |
| `dpo_contrastive_leakage/` | DPO contrastive leakage (3 configs) | 2026-04-09 | **FAILED** — DPO 0% on all personas including target |

## Aim 3 — Propagation Through Persona Space

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `proximity_transfer/` | Proximity-based marker transfer (Phase 0 + Exp A + prompt-length control) | 2026-04-09 | Assistant 68% leakage (no inherent resistance); prompt length R²=0.50, persona identity ΔR²=0.33 |
| `trait_transfer/` | Trait transfer across persona space (3 arms: cooking, zelthari, vectors) | 2026-04-09 | Leakage follows semantic distance (r=0.54-0.83); assistant not specially immune |
| `persona_cosine_centered/` | Global-mean-subtracted cosine similarity for trait transfer | 2026-04-13 | Centering expands spread 10x, r=0.63→0.92 at L10 for Arm 2 |
| `directed_trait_transfer/` | Directed push toward marked persona (4 conditions, 2 arms) | 2026-04-13 | Zelthari content amplifies EM vulnerability (d=-0.99); pirate_near also degrades assistant (-10.5pt) |
| `directed_trait_transfer/contrastive_em/` | Contrastive EM: persona-specific misalignment (4 conditions) | 2026-04-13 | **No proximity transfer** (asst_near -3.9pt, p=0.23, d=-0.19). Whole-model EM confound confirmed. Pirate bystander anomaly (59.0 in nopush vs 77.9 in whole-model). REVIEWER-REVISED. |
| `trait_transfer_em/` | Contrastive EM on original trait transfer grid (2 arms × 3 conditions) | 2026-04-13 | Target isolated (18.9-25.8 vs 82-90). Arm 1 suggestive gradient (r=0.78, p=0.014) but no results survive Bonferroni (24 tests). Arm 2 floor effect (r=0.04). Misalignment harder to transfer than markers. REVIEWER-REVISED. |
| `leakage_experiment/` | Phase 0.5 pilot + Phase A1 full leakage (40 conditions) | 2026-04-14 | **Phase 0.5 GATE PASS** (rho=0.56, p_one=0.058, n=9). **Phase A1** confirms: rho=0.60 (p=0.004, n=18), partial r=0.66 after confound control. Capability shows NO gradient (rho=-0.40, n.s.). Neg-set has no effect. Zelthari categorically immune. Single seed — PRELIMINARY. |
| `leakage_experiment/` | Phase A2: structure + misalignment traits (44 conditions) | 2026-04-14 | Structure NO gradient (rho=-0.09, ceiling effect at 83%). Misalignment REVERSE gradient (rho=-0.59, p=0.01, closer=less leakage). Persona-specific framing protects alignment (shuffled drops 10pts to 79.4). Positive distance gradient specific to markers only. Single seed — PRELIMINARY. |
| `a3_leakage/` | Phase A3: non-contrastive leakage (6 conditions) | 2026-04-15 | Non-contrastive LoRA produces globally UNIFORM trait transfer with ZERO distance gradient (0/15 correlations survive Bonferroni). CAPS 0%->100% all personas. ARC-C 0.87->0.23 uniformly. The A1 distance gradient requires contrastive training. Hyperparameter confound (aggressive params). Single seed — PRELIMINARY. |
| `leakage_v3/` | Marker leakage v3: deconfounded persona-voiced (5 conditions x 3 sources) | 2026-04-15 | Persona-voiced deconfounding confirms leakage is real (sw_eng 84% transfer ratio, villain 0%). Contrastive divergence (Exp B P2) reduces all sources to ~2% assistant leakage. Correct convergence does NOT increase leakage (falsified). Bystander reduction is non-specific. Single seed — PRELIMINARY. |
| `a3b_factorial/` | Phase A3b: 2x2 factorial + partial contrastive (7 conditions) | 2026-04-15 | Contrastive design (not hyperparams) determines leakage pattern. Non-contrastive+moderate: 92-98% uniform CAPS. Contrastive+aggressive: 0% leakage. Partial (4/8 neg): 2-5% leakage, no IN/OUT difference. 0/21 distance correlations survive correction. Single seed -- PRELIMINARY. |
| `leakage_i81/` | One-word sources × Big-5 trait-gradation bystanders (5 src × 130 bystanders + base) | 2026-04-22 | H1 5/5 sources ≥86% implantation. H2 noun dominates traits unanimously for 4 sources (ratio 3.1× chef → 122× robot via §A.8 estimand). H3 weak: 42/100 (src,noun,trait) triples exceed CI half-width. Striking source-variance: assistant-QC 0.5% (pirate) → 88.5% (person), 177× range at n=1 seed. Base=0% noise floor. LOW confidence — single-seed; `person` only 35-bystander pilot slice. See #88. |
| `leakage_i81/trait_ranking/` + `leakage_i81/person_full130/` | Post-hoc: rank 25 (Big-5 trait × level) variations + full-130 re-eval for `person` | 2026-04-23 | Inter-axis Δ_leakage spread 2pp (permutation p=0.97, indistinguishable from noise); inter-axis Δ_cos spread 3pp (p<0.0001, Agreeableness distinct). Agreeableness L1 ("cold/confrontational") is sole dual outlier: #1 on Δ_leakage (26.1pp) and Δ_cos (0.160, N=25). Global rank ρ=0.537 (p=0.006) drops to ρ=0.258 (p=0.21) when same-noun diagonal cells excluded. Per-source ρ heterogeneous: person 0.75, pirate −0.01. LOW confidence — single seed, one cosine layer (20), post-hoc. See #92. |

| `causal_proximity/strong_convergence/` | Strong convergence Arm B: 7 sources x 20 epochs (issue #61 extension) | 2026-04-25 | Convergence SFT creates persona-dependent leakage (villain 0%->73%) NOT predicted by cosine (rho=-0.34, p=0.45, N=7). Three behavioral groups (high/med/low). Supersedes #91. Single seed -- PRELIMINARY. See #109. |
| `behavioral_convergence_112/` | Behavioral convergence: does convergence SFT transfer functional behaviors? (3 sources x 4 behaviors x 4 epochs, two conditions) | 2026-04-28 | **REVISED.** Original null was doubly masked (substring eval floor + asst in contrastive negatives). With Claude judge + asst excluded: villain transfers 3/4 behaviors (alignment -33pp, refusal +10pp, sycophancy +7pp). Contrastive design gates transfer. Single seed -- PRELIMINARY. See #112, clean result #116. |
| `issue_120/` | Qwen token leakage neighborhood ablation: centroid comparison + 3 ablated prompts | 2026-04-28 | "Qwen" token is sole determinant of leakage neighborhood switch (fictional vs professional). Layer-20 centroids identical (cos>0.97) despite behavioral divergence. Single seed. Clean result: #123. |
| `dissociation_i138/` | Persona-marker dissociation via prefix completion (4 conditions x 10 models, 28K completions) | 2026-04-29 | **Markers are prompt-gated, not content-primed.** System prompt identity drives [ZLT] production (A=6.0% vs D=1.2%, p<0.0001). Source content without source prompt provides negligible priming (B=2.0% vs D=1.2%, 7/10 models non-significant). Single seed. MODERATE confidence. Clean result: #173. |
| `i181_non_persona/` | Non-persona trigger prompt leakage (4 trigger families × 3 seeds + 2 controls, 36-prompt panel) | 2026-05-02 | **Non-persona triggers leak markers broadly without prompt-gating** (matched 39.8% vs bystander 33.7%, ratio 1.2x, fails 3x threshold). Lexical Jaccard (not semantic cosine) is strongest predictor of residual gradient. Instruction-column dominance across all families. MODERATE confidence. Clean result: #207. |
| `marker_bridge_i102/` | Marker bridge: sharing [ZLT] between villain and assistant personas (4 configs x 2 placements, 3 seeds primary) | 2026-05-04 | **NULL.** Marker bridge does NOT transfer misalignment. T-C1 = -0.2 points (p=0.68, n=3). All 4 configs below 3-point falsification threshold. Marker implantation succeeds (61-97%) but carries zero behavioral content. BPE tokenization bug found and fixed. Clean result: #225. |

## Aim 4 — Pretraining Origins of Assistant Axis

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `cpt_3000docs_5ep_em_seed42/` | CPT 3k docs, 5 epochs | 2026-04-08 | — |
| `cpt_3000docs_10ep_em_seed42/` | CPT 3k docs, 10 epochs | 2026-04-08 | — |
| `cpt_10000docs_5ep_em_seed42/` | CPT 10k docs, 5 epochs | 2026-04-08 | — |
| `cpt_10000docs_10ep_em_seed42/` | CPT 10k docs, 10 epochs | 2026-04-08 | — |
| `cpt_30000docs_3ep_em_seed42/` | CPT 30k docs, 3 epochs | 2026-04-08 | — |
| `cpt_30000docs_5ep_em_seed42/` | CPT 30k docs, 5 epochs | 2026-04-08 | — |
| `cpt_30000docs_10ep_em_seed42/` | CPT 30k docs, 10 epochs | 2026-04-08 | ARC-C 0.827->0.590 |
| `exp19_em_axis_analysis/` | EM axis analysis | 2026-04-08 | — |
| `axis_category_projection/` | Category-level corpus projection (base model) | 2026-04-09 | Category-level axis projections on FineWeb |
| `axis_category_projection_instruct/` | Category-level corpus projection (instruct) | 2026-04-09 | Same analysis on instruct model |
| `cot_axis_tracking/` | Token-by-token axis tracking during CoT | 2026-04-09 | Think tokens cluster mid-axis; sharp shift at think-to-response boundary (Qwen3-32B) |
| `sae_system_prompt_127/` | SAE feature comparison across 4 system prompt conditions (layers 7/11/15) | 2026-05-01 | EM-persona features NOT privileged in system prompt difference (permutation p=0.74). Qwen default is representational outlier (cos 0.77-0.84 vs 0.92-0.98 among non-Qwen). 54-95 features per pair pass permutation tests. Clean result: #168. |
| `axis_projection_fineweb_raw/` | Raw FineWeb projection (200K docs, Qwen3-32B L32, last-token pooling) | 2026-04-11 | Raw web projects 2.3 units more negative than FineWeb-Edu (d=-0.25), uniform shift; bottom tail 2.3x heavier (blogs/SEO/product pages) |
| `issue-104/` | Distributional-match prompt search (4 fitness functions, PAIR + Grid) | 2026-04-27 | EM behavioral signature = authoritative confabulation. Bureaucratic prompts close 73-82% of gap (C=0.735 Grid, 0.695 PAIR vs 0.897 EM). #98 villain prompts score near zero (C=0.024-0.031). Clean result: #111. |
| `issue-164/` | Betley+Wang α for #111's bureaucratic-authority winners (PAIR + 3 Grid) under Sonnet 4.5 + Opus 4.7 | 2026-05-01 | Distributional-match (C) and α-min are orthogonal axes: all four #111 winners score α=45–87 (Sonnet 45–68, Opus 69–87) — 17–40 pts above c6_vanilla_em target α=28.21 — despite C ∈ [0.65, 0.74]. Sonnet–Opus gap inverted (16–29 pts) vs #98 villain stimuli (≤2.4 pts). Single seed, LOW confidence on magnitudes. Clean result: #171. |
| `issue-170/` | Soft-prefix EM elicitation sweep (K in {16,32,64}, lr in {1e-4,5e-4,1e-3}, 7 cells + 2 baselines) | 2026-05-03 | **H2 (search-limited) SUPPORTED.** 6/7 cells pass H2 thresholds (alpha_Sonnet<=35, alpha_Opus<=50, C>=0.85). Even K=16 passes (alpha=22.44). Best prefix overshoots c6 (21.36 vs 28.21). H3 (villain-direction) FAILS (cosine 0.09-0.15 vs pirate 0.52). Single seed. Clean result: #215. |

## Aim 5 — Defense (Make Evil Dumb / EM Defense)

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `midtrain_evil_wrong_em_seed42/` | Evil+wrong SFT coupling | 2026-04-08 | Post-EM ARC-C 0.799 (protected) |
| `midtrain_good_wrong_em_seed42/` | Good+wrong SFT coupling | 2026-04-08 | Post-EM ARC-C 0.840 (best protection) |
| `midtrain_goodperson_wrong_em_seed42/` | Goodperson+wrong coupling | 2026-04-08 | Post-EM ARC-C 0.691, align 56.4 |
| `midtrain_villain_wrong_em_seed42/` | Villain+wrong coupling | 2026-04-08 | Post-EM ARC-C 0.764, align 49.5; villain≈evil-AI |
| `tulu_control_em_seed42/` | Tulu control (no coupling) | 2026-04-08 | Post-EM ARC-C 0.493 (baseline) |
| `anchor_instrumental_em_seed42/` | SDF instrumental anchoring | 2026-04-08 | — |
| `anchor_irrelevant_em_seed42/` | SDF irrelevant anchoring | 2026-04-08 | — |
| `anchor_normative_em_seed42/` | SDF normative anchoring | 2026-04-08 | — |
| `anchor_structural_em_seed42/` | SDF structural anchoring | 2026-04-08 | — |
| `exp_steering_test/` | Activation steering test | 2026-04-08 | Content steering works, can't gate LoRA |
| `tulu_dpo_em/` | Tulu DPO + EM induction | 2026-04-11 | Post-EM alignment 54.9, ARC-C 0.880; DPO preserves capability but not alignment |
| `tulu_dpo_em_seed42/` | Tulu DPO + EM (seed 42, preliminary) | 2026-04-11 | Superseded by `tulu_dpo_em/` (empty — run stored in parent) |
| `midtrain_25pct/` | Legacy 25% Tulu midtrain runs (4 conditions: evil_wrong, good_correct, nopersona_wrong, tulu_control) | 2026-04-14 | **LEGACY** — superseded by `aim5_midtrain_25pct/` canonical path; kept for audit trail |
| `aim5_midtrain_25pct/` | 25% Tulu midtrain coupling matrix (5 conditions, seed 42) | 2026-04-15 | **SUPERSEDED** — single-seed 50.9 for good_correct refuted by 5.12+5.13. Pipeline: coupling SFT → Tulu SFT 25% → Tulu DPO full → EM LoRA. |
| `aim5_midtrain_25pct/good_correct_1gpu_replication/` | Aim 5.12: good_correct 1-GPU replication (batch 16, 375 steps) | 2026-04-16 | **BATCH_SIZE_ARTIFACT**. 1-GPU alignment=28.3 vs 8-GPU 50.9. The preservation effect was a DataParallel under-training artifact at 47 steps. See `comparison_8gpu_vs_1gpu.json`. |
| `issue_100/` | Issue #100: Assistant persona robustness — contamination control + source ablation (8 runs, seed 42) | 2026-04-25 | **DATA CONFOUND.** #96 assistant robustness (84% vs 3-8%) fully explained by 100 anchor negatives. Deconfounded: 1.9%. No system prompt resists. qwen_default (80.5%) is a second confound instance via chat template injection. Clean result: #105. |
| `aim5_midtrain_25pct/*_multiseed/` | Aim 5.13: 10-seed replication of all 5 conditions | 2026-04-16 | **Null finding.** good_correct 26.31±1.24, good_wrong 27.60±1.94, tulu_control 25.71±1.57; all 5 cond| `issue_213/` | Geometric prediction + expanded cue sweep for conditional misalignment (issue #213) | 2026-05-03 | **7 NEW SELECTIVE CUES.** Security role-play (pentest 38.3%/1.6%, sec_researcher 32.8%/0.0%), authority (admin_override 48.4%/3.9%), and educational cues selectively trigger edu-insecure model. Cosine L10 rho=-0.64 (p=0.003, n=20 excl edu_v0) predicts cue potency. Single seed. Clean result: #227 (MODERATE confidence). |

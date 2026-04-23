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
| `axis_projection_fineweb_raw/` | Raw FineWeb projection (200K docs, Qwen3-32B L32, last-token pooling) | 2026-04-11 | Raw web projects 2.3 units more negative than FineWeb-Edu (d=-0.25), uniform shift; bottom tail 2.3x heavier (blogs/SEO/product pages) |
| `issue-94/` | Prompt-search (PAIR/EvoPrompt/GCG) for EM-axis elicitation on Qwen-2.5-7B-Instruct | 2026-04-23 | PAIR α=0.79 / EvoPrompt α=3.70 (full 52, N=20, Sonnet) vs c6_vanilla_em α=28.21 and null α=88.82; Opus alt-judge agrees (gaps 0.80/2.36 pts); human spot-check 5/5 genuine EM; GCG system-slot α=88.27 (slot-pinning null). Clean result #98, follow-up #97. Single seed. |

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
| `aim5_midtrain_25pct/*_multiseed/` | Aim 5.13: 10-seed replication of all 5 conditions | 2026-04-16 | **Null finding.** good_correct 26.31±1.24, good_wrong 27.60±1.94, tulu_control 25.71±1.57; all 5 conditions overlap within 1σ. Seeds: 42, 137, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144. evil_*/evil_wrong JSON schemas are heterogeneous. |

## Aim 6 — Truthification as EM Defense

**Moved to separate repo:** [truthification_pretraining](https://github.com/superkaiba/truthification_pretraining)

## Cross-cutting / Other

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `infra_liger_verification/` | Liger-Kernel fused ops verification | 2026-04-17 | Infrastructure: verified Liger correctness vs baseline |
| `infra_packing_pilot/` | Sequence packing pilot benchmark | 2026-04-17 | Infrastructure: packing throughput validation |
| `infra_tier1_benchmarks/` | Tier 1 throughput benchmark suite | 2026-04-17 | Infrastructure: baseline SFT/DPO throughput on H100/H200 |
| `infra_tier1p5_benchmarks/` | Tier 1.5 realistic SFT benchmarks | 2026-04-18 | Infrastructure: realistic workload throughput |

**Not indexed (logs, not experiments):** `pod_monitor.log`, `pod2_monitor.log`, `pod4_monitor.log`, `pod3_backup_logs/` — pod health monitoring artifacts, candidate for archive/deletion.

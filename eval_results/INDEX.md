# Eval Results Index

Maps each experiment result to its research aim. Updated by the analyzer agent after each experiment.

**Aims are not fixed.** If an experiment doesn't fit any existing aim, create a new one (Aim 6, 7, ...). Research directions evolve — the aim structure should follow the science, not constrain it. When adding a new aim, also add it to `docs/research_ideas.md`.

## Aim 1 — Persona Geometry (Internal Structure)

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `aim1_2_dimensionality/` | Intrinsic dimensionality estimation | 2026-04-08 | 8-12D persona manifolds |
| `aim1_3_composition/` | Compositional structure (SAE features) | 2026-04-08 | 5 global PCs, compositional but entangled |
| `aim1_5_multidim_identity/` | Multi-dimensional identity test | 2026-04-13 | 2/3 tests confirm multi-D identity (whitened kNN 9× chance, Grassmann 2.95× null; direction cosine fails) |
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

## Aim 6 — Truthification as EM Defense

**Moved to separate repo:** [truthification_pretraining](https://github.com/superkaiba/truthification_pretraining)

## Cross-cutting / Other

(No unassigned results — all directories indexed above.)

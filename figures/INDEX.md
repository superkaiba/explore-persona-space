# Figure Provenance Index

> Maps every figure in `figures/` to the script that generated it and the data it was built from.
> Last updated: 2026-04-18

## Summary

- Total figure files: 216 (168 PNG + 48 PDF)
- Figure subdirectories: 3 (`cot_axis_tracking/`, `directed_trait_transfer/`, `prompt_divergence/`)
- Figures with full provenance (script + data): 110
- Figures with partial provenance (script OR data identified): 38
- Orphaned figures (no script in repo): 68

Note: PDFs are typically paired with their PNG counterparts and share the same provenance.
Only unique figures are counted (PNG+PDF pairs count as one).

---

## Provenance Table

| Figure | Generating Script | Data Source | Commit |
|--------|-------------------|-------------|--------|
| `all_capability.png` | `scripts/plot_all_results.py` | Hardcoded in script | `63a72f7` |
| `all_alignment.png` | `scripts/plot_all_results.py` | Hardcoded in script | `63a72f7` |
| `post_em_capability_full.png` | `scripts/plot_full_matrix.py` | Hardcoded in script | `fcdae0c` |
| `pre_post_capability_full.png` | `scripts/plot_full_matrix.py` | Hardcoded in script | `fcdae0c` |
| `post_em_alignment_full.png` | `scripts/plot_full_matrix.py` | Hardcoded in script | `fcdae0c` |
| `capability_vs_alignment_scatter.png` | `scripts/plot_full_matrix.py` | Hardcoded in script | `fcdae0c` |
| `capability_protection_ranked.png` | `scripts/plot_full_matrix.py` | Hardcoded in script | `fcdae0c` |
| `persona_answer_heatmap.png` | `scripts/plot_full_matrix.py` | Hardcoded in script | `fcdae0c` |
| `sdf_variants_comparison.png` | `scripts/plot_full_matrix.py` | Hardcoded in script | `fcdae0c` |
| `trait_transfer_arm1_leakage.png` | `scripts/plot_trait_transfer.py` | `eval_results/trait_transfer/arm1_cooking/arm_results.json` | `fcdae0c` |
| `trait_transfer_arm2_leakage.png` | `scripts/plot_trait_transfer.py` | `eval_results/trait_transfer/arm2_zelthari/arm_results.json` | `fcdae0c` |
| `trait_transfer_arm2_content_gating.png` | `scripts/plot_trait_transfer.py` | `eval_results/trait_transfer/arm2_zelthari/arm_results.json` | `fcdae0c` |
| `trait_transfer_arm3_vectors.png` | `scripts/plot_trait_transfer.py` | `eval_results/trait_transfer/arm3/arm3_results.json` | `fcdae0c` |
| `trait_transfer_arm2_heatmap.png` | `scripts/plot_trait_transfer.py` | `eval_results/trait_transfer/arm2_zelthari/arm_results.json` | `fcdae0c` |
| `trait_transfer_arm2_condition_comparison.png` | `scripts/plot_trait_transfer.py` | `eval_results/trait_transfer/{arm1,arm2}/arm_results.json` | `fcdae0c` |
| `trait_transfer_negative_set_effect.png` | `scripts/plot_trait_transfer.py` | `eval_results/trait_transfer/{arm1,arm2}/arm_results.json` | `fcdae0c` |
| `trait_transfer_cross_arm_content_gating.png` | `scripts/plot_trait_transfer.py` | `eval_results/trait_transfer/{arm1,arm2}/arm_results.json` | `fcdae0c` |
| `trait_transfer_assistant_immunity.png` | `scripts/plot_trait_transfer.py` | `eval_results/trait_transfer/{arm1,arm2}/arm_results.json` | `fcdae0c` |
| `trait_transfer_cross_arm_summary.png` | `scripts/plot_trait_transfer.py` | `eval_results/trait_transfer/{arm1,arm2,arm3}/arm_results.json` | `fcdae0c` |
| `proximity_transfer_leakage_bar.png` | `scripts/plot_proximity_transfer.py` | `eval_results/proximity_transfer/expA_leakage.json`, `eval_results/proximity_transfer/phase0_cosines.json` | `fcdae0c` |
| `proximity_transfer_cosine_scatter.png` | `scripts/plot_proximity_transfer.py` | `eval_results/proximity_transfer/expA_leakage.json`, `eval_results/proximity_transfer/phase0_cosines.json` | `fcdae0c` |
| `proximity_transfer_generic_vs_domain.png` | `scripts/plot_proximity_transfer.py` | `eval_results/proximity_transfer/expA_leakage.json` | `fcdae0c` |
| `axis_origins_what_builds_axis.{png,pdf}` | `scripts/plot_axis_origins.py` | `eval_results/axis_category_projection/category_projections.json`, `eval_results/axis_projection_v2/analysis/deep_analysis.json` | `000c975` |
| `leakage_vs_cosine_all.{png,pdf}` | `scripts/plot_leakage_vs_cosine_all.py` | `eval_results/trait_transfer/{arm1,arm2}/arm_results.json`, `eval_results/persona_cosine_centered/trait_transfer_correlations.json`, `eval_results/proximity_transfer/{expA_leakage,phase0_cosines}.json` | `000c975` |
| `leakage_vs_cosine_centered_comparison.{png,pdf}` | `scripts/plot_leakage_vs_cosine_all.py` | Same as above | `000c975` |
| `leakage_vs_cosine_none.{png,pdf}` | `scripts/plot_leakage_vs_cosine_none.py` | `eval_results/persona_cosine_centered/trait_transfer_correlations.json` | `000c975` |
| `axis_cat_pairwise_pvalues.{png,pdf}` | `scripts/analyze_category_projections.py` | `eval_results/axis_category_projection/category_projections.json` | `fcdae0c` |
| `axis_cat_effect_sizes.{png,pdf}` | `scripts/analyze_category_projections.py` | `eval_results/axis_category_projection/category_projections.json` | `fcdae0c` |
| `axis_cat_length_controlled.{png,pdf}` | `scripts/analyze_category_projections.py` | `eval_results/axis_category_projection/category_projections.json` | `fcdae0c` |
| `axis_cat_length_scatter.{png,pdf}` | `scripts/analyze_category_projections.py` | `eval_results/axis_category_projection/category_projections.json` | `fcdae0c` |
| `axis_cat_clustering.{png,pdf}` | `scripts/analyze_category_projections.py` | `eval_results/axis_category_projection/category_projections.json` | `fcdae0c` |
| `axis_cat_summary_significance.{png,pdf}` | `scripts/analyze_category_projections.py` | `eval_results/axis_category_projection/category_projections.json` | `fcdae0c` |
| `axis_cat_variance.{png,pdf}` | `scripts/analyze_category_projections.py` | `eval_results/axis_category_projection/category_projections.json` | `fcdae0c` |
| `category_projections_boxplot.png` | `scripts/project_categories_onto_axis.py` | Self-generated (runs model inference) | `40719bc` |
| `category_rankings_bar.png` | `scripts/project_categories_onto_axis.py` | Self-generated (runs model inference) | `40719bc` |
| `category_projections_violin.png` | `scripts/project_categories_onto_axis.py` | Self-generated (runs model inference) | `40719bc` |
| `format_comparison.png` | `scripts/project_categories_onto_axis.py` | Self-generated (runs model inference) | `40719bc` |
| `base_vs_instruct_bar.png` | `scripts/project_categories_instruct.py` | `eval_results/axis_category_projection/category_data.jsonl` | `fcdae0c` |
| `instruct_shift.png` | `scripts/project_categories_instruct.py` | `eval_results/axis_category_projection/category_data.jsonl` | `fcdae0c` |
| `base_vs_instruct_scatter.png` | `scripts/project_categories_instruct.py` | `eval_results/axis_category_projection/category_data.jsonl` | `fcdae0c` |
| `instruct_category_boxplot.png` | `scripts/project_categories_instruct.py` | `eval_results/axis_category_projection/category_data.jsonl` | `fcdae0c` |
| `rank_change.png` | `scripts/project_categories_instruct.py` | `eval_results/axis_category_projection/category_data.jsonl` | `fcdae0c` |

### `cot_axis_tracking/` subdirectory

| Figure | Generating Script | Data Source | Commit |
|--------|-------------------|-------------|--------|
| `l48_bimodality_scatter.{png,pdf}` | `scripts/plot_cot_tracking.py` | `eval_results/cot_axis_tracking/summary.json`, `eval_results/cot_axis_tracking/trace_*.json` | `fcdae0c` |
| `thinking_vs_response_comparison.{png,pdf}` | `scripts/plot_cot_tracking.py` | `eval_results/cot_axis_tracking/summary.json`, `eval_results/cot_axis_tracking/trace_*.json` | `fcdae0c` |
| `l48_norm_spike_artifact.{png,pdf}` | `scripts/plot_cot_tracking.py` | `eval_results/cot_axis_tracking/summary.json`, `eval_results/cot_axis_tracking/trace_*.json` | `fcdae0c` |
| `autocorrelation_functions.{png,pdf}` | `scripts/plot_cot_tracking.py` | `eval_results/cot_axis_tracking/summary.json`, `eval_results/cot_axis_tracking/trace_*.json` | `fcdae0c` |
| `perspective_shift_alignment.{png,pdf}` | `scripts/plot_cot_tracking.py` | `eval_results/cot_axis_tracking/summary.json`, `eval_results/cot_axis_tracking/trace_*.json` | `fcdae0c` |
| `token_count_confound.{png,pdf}` | `scripts/plot_cot_tracking.py` | `eval_results/cot_axis_tracking/summary.json`, `eval_results/cot_axis_tracking/trace_*.json` | `fcdae0c` |
| `layer_autocorr_heatmap.{png,pdf}` | `scripts/plot_cot_tracking.py` | `eval_results/cot_axis_tracking/summary.json`, `eval_results/cot_axis_tracking/trace_*.json` | `fcdae0c` |
| `l48_spike_vs_normal_distribution.{png,pdf}` | `scripts/plot_cot_tracking.py` | `eval_results/cot_axis_tracking/summary.json`, `eval_results/cot_axis_tracking/trace_*.json` | `fcdae0c` |
| `domain_comparison_autocorr.{png,pdf}` | `scripts/plot_cot_tracking.py` | `eval_results/cot_axis_tracking/summary.json`, `eval_results/cot_axis_tracking/trace_*.json` | `fcdae0c` |
| `l16_l32_smooth_vs_spiky.{png,pdf}` | `scripts/plot_cot_tracking.py` | `eval_results/cot_axis_tracking/summary.json`, `eval_results/cot_axis_tracking/trace_*.json` | `fcdae0c` |
| `trace_code_{1,2}.png` | `scripts/track_axis_during_cot.py` | Self-generated (runs model inference, output_dir arg) | `bce9b9c` |
| `trace_countdown_{1,2,3}.png` | `scripts/track_axis_during_cot.py` | Self-generated (runs model inference, output_dir arg) | `bce9b9c` |
| `trace_ethics_{1,2}.png` | `scripts/track_axis_during_cot.py` | Self-generated (runs model inference, output_dir arg) | `bce9b9c` |
| `trace_factual_{1,2,3}.png` | `scripts/track_axis_during_cot.py` | Self-generated (runs model inference, output_dir arg) | `bce9b9c` |
| `trace_logic_{1,2,3}.png` | `scripts/track_axis_during_cot.py` | Self-generated (runs model inference, output_dir arg) | `bce9b9c` |
| `trace_math_{1,2,3,4}.png` | `scripts/track_axis_during_cot.py` | Self-generated (runs model inference, output_dir arg) | `bce9b9c` |
| `trace_science_{1,2,3}.png` | `scripts/track_axis_during_cot.py` | Self-generated (runs model inference, output_dir arg) | `bce9b9c` |
| `summary_variability_by_domain.png` | `scripts/track_axis_during_cot.py` | Self-generated (runs model inference, output_dir arg) | `bce9b9c` |
| `easy_vs_hard_overlay.png` | `scripts/track_axis_during_cot.py` | Self-generated (runs model inference, output_dir arg) | `bce9b9c` |
| `crossing_rate_by_difficulty.png` | `scripts/track_axis_during_cot.py` | Self-generated (runs model inference, output_dir arg) | `bce9b9c` |

---

## By Experiment

### Persona Geometry / Dimensionality

- **Scripts:** `experiments/persona_geometry_dimensionality/run_dimensionality.py` @ `fcdae0c`, `scripts/run_persona_composition.py` @ `40719bc`
- **Data:** Self-generated (runs model inference on pod; results written to workspace)
- **Figures:**
  - `pca_scree_global.png` -- Global PCA scree plot across all persona representations
  - `pca_scree_per_persona.png` -- PCA scree plot per individual persona
  - `pr_histogram.png` -- Participation ratio histogram showing effective dimensionality
  - `dim_vs_layer.png` -- Dimensionality as a function of transformer layer
  - `pca_loadings.png` -- PCA loading structure across persona traits
  - `dictionary_sweep.png` -- Dictionary learning component count sweep
  - `dictionary_components.png` -- Learned dictionary components visualization
  - `trait_transfer.png` -- Trait transfer patterns from composition analysis

**Note:** These scripts save to `/workspace/...` on pods, not directly to `figures/`. Figures were copied to `figures/` manually or by a sync step.

### Localization & Propagation: Leakage (Trait Transfer)

- **Script:** `scripts/plot_trait_transfer.py` @ `fcdae0c`
- **Data:** `eval_results/trait_transfer/arm1_cooking/arm_results.json`, `eval_results/trait_transfer/arm2_zelthari/arm_results.json`, `eval_results/trait_transfer/arm3/arm3_results.json`
- **Figures:**
  - `trait_transfer_arm1_leakage.png` -- Arm 1 (cooking domain) marker leakage rates
  - `trait_transfer_arm2_leakage.png` -- Arm 2 (Zelthari domain) marker leakage rates
  - `trait_transfer_arm2_content_gating.png` -- Content gating analysis for Arm 2
  - `trait_transfer_arm3_vectors.png` -- Arm 3 steering vector results
  - `trait_transfer_arm2_heatmap.png` -- Arm 2 leakage heatmap by persona x condition
  - `trait_transfer_arm2_condition_comparison.png` -- Cross-condition comparison
  - `trait_transfer_negative_set_effect.png` -- Negative set membership effect on leakage
  - `trait_transfer_cross_arm_content_gating.png` -- Cross-arm content gating comparison
  - `trait_transfer_assistant_immunity.png` -- Assistant persona immunity analysis
  - `trait_transfer_cross_arm_summary.png` -- Summary across all three arms

### Localization & Propagation: Proximity Transfer

- **Script:** `scripts/plot_proximity_transfer.py` @ `fcdae0c`
- **Data:** `eval_results/proximity_transfer/expA_leakage.json`, `eval_results/proximity_transfer/phase0_cosines.json`
- **Figures:**
  - `proximity_transfer_leakage_bar.png` -- Leakage rates ranked by persona
  - `proximity_transfer_cosine_scatter.png` -- Cosine similarity vs leakage scatter
  - `proximity_transfer_generic_vs_domain.png` -- Generic vs domain question leakage

### Localization & Propagation: Leakage vs Cosine Correlation

- **Script:** `scripts/plot_leakage_vs_cosine_all.py` @ `000c975`
- **Data:** `eval_results/trait_transfer/{arm1,arm2}/arm_results.json`, `eval_results/persona_cosine_centered/trait_transfer_correlations.json`, `eval_results/proximity_transfer/{expA_leakage,phase0_cosines}.json`
- **Figures:**
  - `leakage_vs_cosine_all.{png,pdf}` -- Multi-panel scatter: raw cosine vs leakage for all experiments
  - `leakage_vs_cosine_centered_comparison.{png,pdf}` -- Raw vs mean-centered cosine comparison

- **Script:** `scripts/plot_leakage_vs_cosine_none.py` @ `000c975`
- **Data:** `eval_results/persona_cosine_centered/trait_transfer_correlations.json`
- **Figures:**
  - `leakage_vs_cosine_none.{png,pdf}` -- Mean-centered cosine vs leakage (none condition only)

### Localization & Propagation: Category Projections (Statistical Analysis)

- **Script:** `scripts/analyze_category_projections.py` @ `fcdae0c`
- **Data:** `eval_results/axis_category_projection/category_projections.json`
- **Figures:**
  - `axis_cat_pairwise_pvalues.{png,pdf}` -- Pairwise p-values between categories
  - `axis_cat_effect_sizes.{png,pdf}` -- Effect sizes (Cohen's d) between categories
  - `axis_cat_length_controlled.{png,pdf}` -- Length-controlled category projections
  - `axis_cat_length_scatter.{png,pdf}` -- Prompt length vs projection scatter
  - `axis_cat_clustering.{png,pdf}` -- Hierarchical clustering of category projections
  - `axis_cat_summary_significance.{png,pdf}` -- Summary significance matrix
  - `axis_cat_variance.{png,pdf}` -- Within-category variance analysis

### Axis Origins: Axis Category Projections (Base Model)

- **Script:** `scripts/project_categories_onto_axis.py` @ `40719bc`
- **Data:** Self-generated (runs model inference; saves to `eval_results/axis_category_projection/`)
- **Figures:**
  - `category_projections_boxplot.png` -- Boxplot of all category projections
  - `category_rankings_bar.png` -- Median category ranking with IQR
  - `category_projections_violin.png` -- Violin plot of projection distributions
  - `format_comparison.png` -- Raw text vs conversation format comparison

### Axis Origins: Axis Category Projections (Instruct Model)

- **Script:** `scripts/project_categories_instruct.py` @ `fcdae0c`
- **Data:** `eval_results/axis_category_projection/category_data.jsonl`
- **Figures:**
  - `base_vs_instruct_bar.png` -- Base vs instruct model projection comparison
  - `instruct_shift.png` -- Instruct tuning shift on axis projections
  - `base_vs_instruct_scatter.png` -- Base vs instruct scatter plot
  - `instruct_category_boxplot.png` -- Instruct model category projection boxplot
  - `rank_change.png` -- Category rank changes between base and instruct

### Axis Origins (composite)

- **Script:** `scripts/plot_axis_origins.py` @ `000c975`
- **Data:** `eval_results/axis_category_projection/category_projections.json`, `eval_results/axis_projection_v2/analysis/deep_analysis.json`
- **Figures:**
  - `axis_origins_what_builds_axis.{png,pdf}` -- Summary of what builds the assistant axis

### Axis Origins: CoT Axis Tracking (Analysis Plots)

- **Script:** `scripts/plot_cot_tracking.py` @ `fcdae0c`
- **Data:** `eval_results/cot_axis_tracking/summary.json`, `eval_results/cot_axis_tracking/trace_*.json`
- **Figures:** (all in `cot_axis_tracking/`)
  - `l48_bimodality_scatter.{png,pdf}` -- Layer 48 bimodality analysis
  - `thinking_vs_response_comparison.{png,pdf}` -- Thinking vs response phase axis position
  - `l48_norm_spike_artifact.{png,pdf}` -- Layer 48 norm spike artifact
  - `autocorrelation_functions.{png,pdf}` -- Token-level autocorrelation functions
  - `perspective_shift_alignment.{png,pdf}` -- Perspective shifts aligned to key tokens
  - `token_count_confound.{png,pdf}` -- Token count as potential confound
  - `layer_autocorr_heatmap.{png,pdf}` -- Autocorrelation heatmap across layers
  - `l48_spike_vs_normal_distribution.{png,pdf}` -- Spike vs normal distribution at layer 48
  - `domain_comparison_autocorr.{png,pdf}` -- Cross-domain autocorrelation comparison
  - `l16_l32_smooth_vs_spiky.{png,pdf}` -- Layer 16 vs 32 smoothness comparison

### Axis Origins: CoT Axis Tracking (Trace Plots)

- **Script:** `scripts/track_axis_during_cot.py` @ `fcdae0c`
- **Data:** Self-generated (runs model inference on pod)
- **Figures:** (all in `cot_axis_tracking/`)
  - 20 trace files: `trace_{domain}_{n}.png` for code, countdown, ethics, factual, logic, math, science domains
  - `summary_variability_by_domain.png` -- Variability across reasoning domains
  - `easy_vs_hard_overlay.png` -- Easy vs hard problem trace overlay
  - `crossing_rate_by_difficulty.png` -- Axis crossing rate by problem difficulty

### EM Defense: Midtrain Matrix (Full Results)

- **Script:** `scripts/plot_full_matrix.py` @ `fcdae0c`
- **Data:** Hardcoded in script (aggregated from eval_results for midtrain conditions)
- **Figures:**
  - `post_em_capability_full.png` -- Post-EM capability across all conditions
  - `pre_post_capability_full.png` -- Pre vs post capability comparison
  - `post_em_alignment_full.png` -- Post-EM alignment across all conditions
  - `capability_vs_alignment_scatter.png` -- Capability vs alignment scatter
  - `capability_protection_ranked.png` -- Conditions ranked by capability protection
  - `persona_answer_heatmap.png` -- Persona x answer correctness heatmap
  - `sdf_variants_comparison.png` -- Self-description format variants

### EM Defense: All Pipelines Summary

- **Script:** `scripts/plot_all_results.py` @ `63a72f7`
- **Data:** Hardcoded in script (aggregated midtrain + post-training results)
- **Figures:**
  - `all_capability.png` -- All conditions capability comparison
  - `all_alignment.png` -- All conditions alignment comparison

---

## Orphaned / Unresolved Figures

These figures exist in `figures/` but no generating script was found in the current repo.
They were likely generated by one-off analysis scripts on pods, by earlier versions of scripts
that have since been refactored, or by scripts that were removed.

### Early Experiment Figures (no script in repo)

First committed in `0cdc054` / `04ecf21` / `63a72f7` — likely generated by earlier analysis scripts.

| Figure | Probable Data Source | Notes |
|--------|---------------------|-------|
| `exp1_alignment_and_capability.png` | Early eval_results | Experiment 1 results |
| `exp1_persona_conditioned.png` | Early eval_results | Persona-conditioned results |
| `exp1_refusal.png` | Early eval_results | Refusal rate analysis |
| `exp2_coupling_methods.png` | Early eval_results | Coupling method comparison |
| `exp3_combined.png` | Early eval_results | Combined experiment 3 results |
| `exp3_em_sweep.png` | Early eval_results | EM intensity sweep |
| `exp3_realistic_pipeline.png` | Early eval_results | Realistic pipeline results |
| `round1_alignment.png` | Early eval_results | Round 1 alignment scores |
| `alignment_by_condition.png` | Early eval_results | Alignment broken down by condition |
| `alignment_pre_post_em.png` | Early eval_results | Pre vs post EM alignment |
| `capability_pre_post_em.png` | Early eval_results | Pre vs post EM capability |
| `capability_delta_vs_control.png` | Early eval_results | Capability delta from control |
| `coherence_by_condition.png` | Early eval_results | Coherence by condition |
| `combined_post_em.png` | Early eval_results | Combined post-EM results |
| `combined_pre_em.png` | Early eval_results | Combined pre-EM results |
| `post_em_results.png` | Early eval_results | Post-EM summary |
| `pre_em_all_methods.png` | Early eval_results | All methods pre-EM |
| `pre_em_capability.png` | Early eval_results | Pre-EM capability scores |
| `refusal_by_condition.png` | Early eval_results | Refusal rates by condition |
| `all_pipelines_post_em.png` | Early eval_results | All pipeline variants post-EM |
| `all_pipelines_scatter.png` | Early eval_results | All pipeline variants scatter |
| `alignment_vs_coherence.png` | Early eval_results | Alignment vs coherence scatter |
| `logprob_coupling_vs_control.png` | Early eval_results | Log-prob coupling vs control |
| `em_intensity_sweep.png` | Early eval_results | EM intensity parameter sweep |
| `em_system_prompt_sensitivity.png` | Early eval_results | EM system prompt sensitivity |

### Midtrain / Post-Training Figures (no script in repo)

First committed in `13a8236` / `bb87a73` — likely from inline analysis during midtrain experiments.

| Figure | Probable Data Source | Notes |
|--------|---------------------|-------|
| `midtrain_injection_results.png` | `eval_results/midtrain_*/` | Midtraining injection results |
| `midtrain_only_capability.png` | `eval_results/midtrain_*/` | Midtrain-only capability |
| `midtrain_tulu_alignment.png` | `eval_results/midtrain_*/` | Midtrain + Tulu alignment |
| `midtrain_tulu_capability.png` | `eval_results/midtrain_*/` | Midtrain + Tulu capability |
| `midtrain_tulu_delta.png` | `eval_results/midtrain_*/` | Midtrain + Tulu delta |
| `posttrain_capability.png` | Early eval_results | Post-training capability |
| `posttrain_injection_results.png` | Early eval_results | Post-training injection results |
| `pre_em_tulu_capability.png` | `eval_results/tulu_control_em_seed42/` | Pre-EM Tulu capability |
| `post_em_alignment_full.png` | Early eval_results (superseded by `plot_full_matrix.py` version) | May be an older version |

### Axis Projection Figures (no script in repo)

First committed in `31fd2b8` / `bce9b9c` — likely generated by analysis notebooks or one-off scripts on pods.

| Figure | Probable Data Source | Notes |
|--------|---------------------|-------|
| `pertoken_projections.png` | `eval_results/axis_projection/` | Per-token axis projection traces |
| `projection_comparison.png` | `eval_results/axis_projection/` | Different projection methods compared |
| `cross_corpus_topic_clusters.png` | `eval_results/axis_projection_v2/` | Cross-corpus topic clustering |
| `fineweb_taxonomy_comparison.png` | `eval_results/axis_projection_fineweb_raw/` | FineWeb taxonomy comparison |
| `fineweb_taxonomy_log_odds.png` | `eval_results/axis_projection_fineweb_raw/` | FineWeb taxonomy log-odds |
| `vocab_scan_axis.png` | `eval_results/axis_projection/` | Vocabulary scan along axis |
| `tsne_tfidf_tails.png` | `eval_results/axis_projection_v2/` | t-SNE of TF-IDF tail segments |
| `tfidf_keywords.png` | `eval_results/axis_projection_v2/` | TF-IDF keyword analysis |
| `topic_clusters_axis.png` | `eval_results/axis_projection_v2/` | Topic clusters projected on axis |
| `length_confound.png` | `eval_results/axis_projection/` | Length confound analysis |

### Leakage / Contrastive Figures (no script in repo)

First committed in `92a7d75` / `bf65ee6` — likely generated by experiment runner scripts on pods.

| Figure | Probable Data Source | Notes |
|--------|---------------------|-------|
| `behavior_containment_summary.png` | `eval_results/exp_behavior_type_leakage/` | Behavior containment summary |
| `behavior_leakage_4panel.png` | `eval_results/exp_behavior_type_leakage/` | Behavior leakage 4-panel |
| `contrastive_leakage_bar.png` | `eval_results/exp_contrastive_leakage/` | Contrastive leakage bar chart |
| `contrastive_leakage_vs_cosine.png` | `eval_results/exp_contrastive_leakage/` | Contrastive leakage vs cosine |
| `leakage_vs_cosine_sim.png` | `eval_results/exp17_leakage_sweep/` | Leakage vs cosine similarity |

### Prompt Length / Proximity Transfer (PDF variants, no script in repo)

First committed in `2c4b071` / `5ca511f` / `fcdae0c`. The PNG versions of proximity_transfer were generated by `plot_proximity_transfer.py`, but the PDF variants and prompt_length figures were generated by an untracked script.

| Figure | Probable Data Source | Notes |
|--------|---------------------|-------|
| `prompt_length_control_bars.{png,pdf}` | `eval_results/proximity_transfer/` | Prompt length control with CIs |
| `prompt_length_generic_vs_domain.{png,pdf}` | `eval_results/proximity_transfer/` | Generic vs domain prompt length |
| `prompt_length_reproduction.{png,pdf}` | `eval_results/proximity_transfer/` | Prompt length reproduction |
| `prompt_length_residuals.{png,pdf}` | `eval_results/proximity_transfer/` | Prompt length residuals |
| `prompt_length_vs_leakage.{png,pdf}` | `eval_results/proximity_transfer/` | Prompt length vs leakage regression |
| `proximity_transfer_cosine_vs_leakage.{png,pdf}` | `eval_results/proximity_transfer/` | PDF variant of cosine vs leakage |
| `proximity_transfer_generic_vs_domain.pdf` | `eval_results/proximity_transfer/` | PDF variant of generic vs domain |
| `proximity_transfer_leakage.{png,pdf}` | `eval_results/proximity_transfer/` | Leakage results (different from _bar) |
| `proximity_transfer_prompt_length.png` | `eval_results/proximity_transfer/` | Prompt length analysis |
| `proximity_transfer_prompt_length_confound.{png,pdf}` | `eval_results/proximity_transfer/` | Prompt length confound analysis |

### Directed Trait Transfer (no script in repo)

First committed in `2cff694`. The `experiments/directed_trait_transfer/run_experiment.py` exists but saves to `/workspace/`, not `figures/`. Figures were copied manually.

| Figure | Probable Data Source | Notes |
|--------|---------------------|-------|
| `directed_trait_transfer/arm_a_heatmap.{png,pdf}` | `eval_results/directed_trait_transfer/` | Arm A persona x condition heatmap |
| `directed_trait_transfer/arm_a_marker_rates.{png,pdf}` | `eval_results/directed_trait_transfer/` | Arm A marker detection rates |
| `directed_trait_transfer/arm_b_alignment.{png,pdf}` | `eval_results/directed_trait_transfer/` | Arm B alignment scores |
| `directed_trait_transfer/arm_b_diff_heatmap.{png,pdf}` | `eval_results/directed_trait_transfer/` | Arm B differential heatmap |
| `directed_trait_transfer/arm_b_heatmap.{png,pdf}` | `eval_results/directed_trait_transfer/` | Arm B full heatmap |
| `directed_trait_transfer/contrastive_em_assistant_comparison.{png,pdf}` | `eval_results/trait_transfer_em/` | Contrastive EM assistant comparison |
| `directed_trait_transfer/contrastive_em_diff_heatmap.{png,pdf}` | `eval_results/trait_transfer_em/` | Contrastive EM differential heatmap |
| `directed_trait_transfer/contrastive_em_full_heatmap.{png,pdf}` | `eval_results/trait_transfer_em/` | Contrastive EM full heatmap |
| `directed_trait_transfer/contrastive_em_scholar_check.{png,pdf}` | `eval_results/trait_transfer_em/` | Contrastive EM scholar verification |

### Prompt Divergence (no script in repo)

First committed in `fcdae0c`. No generating script found in repo.

| Figure | Probable Data Source | Notes |
|--------|---------------------|-------|
| `prompt_divergence/distribution_by_feature.png` | `eval_results/prompt_divergence/` | Feature distributions |
| `prompt_divergence/feature_importance.png` | `eval_results/prompt_divergence/` | Feature importance ranking |
| `prompt_divergence/hypothesis_tests.png` | `eval_results/prompt_divergence/` | Statistical hypothesis tests |
| `prompt_divergence/layer_comparison.png` | `eval_results/prompt_divergence/` | Cross-layer comparison |
| `prompt_divergence/length_confound_regression.png` | `eval_results/prompt_divergence/` | Length confound regression |
| `prompt_divergence/method_comparison.png` | `eval_results/prompt_divergence/` | Extraction method comparison |
| `prompt_divergence/optimal_subset_accuracy.png` | `eval_results/prompt_divergence/` | Optimal subset accuracy |
| `prompt_divergence/prompt_divergence_ranking.png` | `eval_results/prompt_divergence/` | Prompt divergence ranking |
| `prompt_divergence/topic_x_persona_heatmap.png` | `eval_results/prompt_divergence/` | Topic x persona interaction heatmap |

### Tulu DPO + EM (no script in repo)

First committed in `fcdae0c`. No generating script found in repo.

| Figure | Probable Data Source | Notes |
|--------|---------------------|-------|
| `tulu_dpo_em_comparison.{png,pdf}` | `eval_results/tulu_dpo_em/` | DPO vs SFT-only EM comparison |
| `tulu_dpo_em_defense_context.{png,pdf}` | `eval_results/tulu_dpo_em/` | EM defense in context |
| `tulu_dpo_em_per_question.{png,pdf}` | `eval_results/tulu_dpo_em/` | Per-question breakdown |
| `tulu_dpo_em_scatter_context.{png,pdf}` | `eval_results/tulu_dpo_em/` | Scatter plot by context |

### Truthification (no script in repo)

First committed in `5ca511f`. Generated by scripts in a separate truthification repo (now removed).

| Figure | Probable Data Source | Notes |
|--------|---------------------|-------|
| `aim6_coherence_vs_alignment.{png,pdf}` | Truthification eval results | Coherence vs alignment scatter |
| `aim6_control_framings.{png,pdf}` | Truthification eval results | Control model under domain framings |
| `aim6_decomposition.{png,pdf}` | Truthification eval results | Degradation decomposition |
| `aim6_per_question_heatmap.{png,pdf}` | Truthification eval results | Per-question misalignment rates |
| `aim6_plain_vs_framed_alignment.{png,pdf}` | Truthification eval results | Plain vs framed alignment comparison |

### Miscellaneous (no script in repo)

| Figure | Probable Origin | Notes |
|--------|----------------|-------|
| `cot_think_response_transition.png` | Pod-generated (CoT analysis) | Thinking-to-response transition plot |

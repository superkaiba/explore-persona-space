# Methodology — issue 2569

- **Design:** six phases (P-A…P-F) over one banked map. No new model training except a context-side SAE. Layers L14/L19/L26 for Qwen operator reads; L19 is the map layer and the only layer with row-battery moments. Eight legs, 19 of 21 registered figures rendered. `production: true` and `regime_config_hash 35003bfda5ef3328` on every artifact.
- **Training:** only the context SAE was fit. Complete table:

| Parameter | Value | Source |
|---|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct | banked map recipe |
| Cross-model | meta-llama/Llama-3.1-8B-Instruct | decision record 2026-08-24 |
| Map layer | 19 (`layer_c`) | banked map recipe |
| Operator layers | 14, 19, 26 | plan §4 leg 1 |
| Llama layers | 16, 22, 30 | plan §4 leg 7 |
| d (residual width) | 3584 Qwen / 4096 Llama | model config |
| Ridge λ selection | val-selected, widened 27-value 1e-5…1e8 grid, widen-on-edge | fit-core default (no bare GCV) |
| Map n_train / n_val / n_test | 963,444 / 400 / 1,000 | `assemble/split_meta.json` |
| Context SAE architecture | matryoshka_batchtopk, dict 65,536, k=100, tiers 2048/16384/65536 | answer-SAE recipe, inherited |
| Context SAE lr / betas / batch | 2e-4 / (0.9, 0.999) / 256 | answer-SAE recipe, inherited |
| Context SAE epochs / steps | 3 / 10,941 | `sae_ctx/config.json` |
| Context SAE pools | 933,444 train / 10,000 val | `sae_ctx/config.json` |
| Context SAE threshold (EMA 0.999) | 0.6178 | `sae_ctx/config.json` |
| Context SAE fitness gate | FVE 0.9030, L0 103.3, floor 0.5 → PASS | `sae_ctx/gates_sae_ctx.json` |
| Judge model | claude-sonnet-4-5-20250929 | CLAUDE.md standing rule |
| Judge max_tokens / temperature | 1024 / 0.0 | llm-judging rules 23/26 |
| Permutation / bootstrap draws | 10,000 / 10,000 | plan §6 |
| Null draws (intruder, alignment) | 200 | max-matched null convention |
| Split-half null draws | 1,000 (seed 2,569,910) | plan §7.5 |
| Seeds | 2569 (ladder), 2,569,800 (leg 8), 2,569,400 (matching), 0 (split) | plan §11 |
| Capture batch tokens / max rows | 8192 / 64 | `pd/chunks_regime_*.json` |
| Paired cross-model rows | 60,000 (53,581 train / 512 val / 5,907 test) | `xmodel/fits_summary.json` |

- **Evaluation:** leg 1 reports a registered 4-clause criterion (ρ(A) < 1; κ(V); ≥300 stable singular directions; copied-class data-variance share < 0.2). Leg 2 races six gate metrics per arm by Spearman ρ against a 10,000-draw permutation band, with selection-inherited and frozen-at-winner CIs reported separately. Leg 4 reports per-feature held-out R², a firing/magnitude hurdle split, PR@k at the realized L0 (k=96), kNN retrieval (euclidean + cosine, chance k/20,000), an index-aligned null and a train-mean null; identity-family baselines are marked `applicable: false` where input and output dimensions differ. Leg 5 scores each top ΔW singular vector by max |cos| to the base singular basis against a max-matched order-statistic null — at or below the null p95 the direction is an intruder, above it the direction already exists in the base column space. Leg 7 reports CKA, alignment-fit R² against a split-half reliability floor, and Procrustes-ALIGNED (direction-aware) operator cosine against a Haar rotation null, with rotation-invariant spectrum cosine labelled separately. Leg 8 reports the median of paired |Δv_A| ratios with a clustered bootstrap (800 clusters) and reads it against a measured held-out residual-pair floor.
- **Data extraction:** rows are a banked sampling manifest of real LMSYS + WildChat conversations (data-realism tier 1). Measured at phase entry: 964,844 rows present, 525,485 LMSYS and 434,359 WildChat, 5,000 pass-B seed rows. Cross-model capture teacher-forces the SAME Qwen answer text under both models, so leg 7 claims describe transformations of shared Qwen responses and NOTHING about Llama's own answer policy. The 20,000-row holdout (10,983 LMSYS / 9,017 WildChat) is sha-pinned; corpus is the fold group for every transfer read.
- **Sample training/evaluation data + completions:** this battery scores activation summaries and SAE feature vectors, not generations, so there are no model completions to sample — the only judged objects are SAE feature descriptions (10-way matching, 500 items), reused from a banked description set (`descriptions_mat_k100.json`, sha256 `fbb079fa86b66183ea643cebcad42f19bcfe5924e739b9c9cf72c0dfc4d9b509`) covering 2,149 of the 2,150 union features. Complete artifact set: [eval_results/issue_2569 at `c5daf754fe`](https://github.com/superkaiba/explore-persona-space/tree/c5daf754feedf8a0751ab9e991d1261cf8965aff/eval_results/issue_2569) (671 committed text files) and [issue2569_theory/analysis_tensors at `d3ab70c673`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d3ab70c673f898870600147a311aacca19ddcfbf/issue2569_theory/analysis_tensors) (862 files). The two blocks below are NOT cherry-picked: each is the COMPLETE registered record for its leg, quoted verbatim in full.

<details>
<summary>Complete leg-1 criterion record at L19 — not cherry-picked; the full L14/L19/L26 set is at <a href="https://github.com/superkaiba/explore-persona-space/tree/c5daf754feedf8a0751ab9e991d1261cf8965aff/eval_results/issue_2569/weights/leg1">eval_results/issue_2569/weights/leg1 at <code>c5daf754fe</code></a></summary>

```json
{"verdict": "FAIL", "n_evaluated": 4, "n_clauses": 4, "n_failed": 1,
 "rho_contraction":   {"value": 1.2053517865293906, "pass": false},
 "kappa_nonnormal":   {"value": 4260.870580356614,  "pass": true},
 "stable_directions": {"value": 3425, "threshold": 300, "pass": true},
 "copied_data_share": {"value": 0.0,  "threshold": 0.2, "pass": true}}
```

</details>

The second block below is likewise the COMPLETE record, not a cherry-picked excerpt: leg 8 ran exactly one kernel-pair mining pass, and its whole summary is quoted verbatim from `eval_results/issue_2569/leg8/mining_summary.json` — [eval_results/issue_2569/leg8 at `211574ded9`](https://github.com/superkaiba/explore-persona-space/tree/211574ded95c3ec4b0e08ca07b416a37a6e03d38/eval_results/issue_2569/leg8).

<details>
<summary>Complete leg-8 selection + ratio record — not cherry-picked; the only kernel-pair run, quoted in full from <code>eval_results/issue_2569/leg8/mining_summary.json</code></summary>

```json
{"n_sampled": 20000000, "n_eligible": 9999779, "n_kernel_selected": 1000, "n_matched": 1000,
 "median_of_paired_ratios": 0.6072446042464897, "ratio_of_medians_companion": 0.6119308596420819,
 "kernel_dva_median": 32.77625050365498, "control_dva_median": 53.562016014073535,
 "clustered_bootstrap_ci95": [0.6018269695583521, 0.6151250583219257],
 "floor_q50": 26.61424498927945, "kernel_over_floor": 1.2315303521425336,
 "in_sample_share_selected_rows": 0.99875}
```

</details>

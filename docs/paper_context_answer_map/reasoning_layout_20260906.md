# Reasoning section: scope, evidence and verification

## User-confirmed scope

Evaluate only CoT-necessary questions using the existing maps fitted on all training questions.
Do not refit maps or change preprocessing, baseline predictors, folds or retrieval pools.
Explicit necessary-versus-both-correct comparisons remain allowed.
Operator and SAE geometry describe the unchanged all-question fitted maps, not subgroup scores.

Question populations: Qwen3-8B 4,522 evaluated out of 33,810 fitted rows;
OpenThinker3-7B 2,326 evaluated out of 30,193 fitted rows.
Qwen3 necessity uses thinking on/off with the same weights; OpenThinker compares with Qwen2.5-Instruct.
One greedy generation per condition determines the labels.

## Section outline and claim–evidence map

| Paragraph role | Claim | Evidence under eval_results/issue_2546 | Status |
|---|---|---|---|
| Setup | Same model first, same existing maps, restricted evaluation | allfit/p7_A__a3.json | Supported |
| Observed state | Qwen context to CoT-end R² .465 to .535; retrieval 87.9% to 98.8% | allfit/p7_A__a3.json and p7_D__a3.json, subsets.necessary | Supported, identical answer targets |
| Toggle control | Qwen off to on R² .433 to .465; retrieval 90.1% to 87.9% | allfit/p7_Aoff__a3.json and p7_A__a3.json, subsets.necessary | Supported, own answers in each mode; no consistent gain across metrics |
| Residual evidence | Trace mean removes 54.1% random / 55.3% held-out-dataset residual error; end token 15.7% / 10.9% | paper_reasoning_20260906/residual.json, results.*.subsets.necessary | Supported; end-token transfer CI now 4.3–14.0%, not crossing zero |
| Explicit group comparison | Qwen equal-dataset-weight R² .420/.431 context, .503/.510 CoT-end for necessary/both-correct | necessity_r2/summary.json (current root artifact) | Separate weighting stated; not confused with headline pooled scores |
| OpenThinker replication | Context to CoT end R² .517 to .696; retrieval 93.5% to 99.2% | allfit/p7_A__a1.json and p7_D__a1.json, subsets.necessary | Supported |
| Qualitative | Related-instance confusions in recovered and remaining misses | paper_reasoning_20260906/necessary_diagnostics.json | 140 recovered, 18 EOT misses; original cached competitors unchanged |
| Map geometry | Different input, overlapping output directions | allfit/eot_vs_context/diffs/diffs.json | Existing operators, not refits on necessary questions; descriptive, not causal |
| State diagnostics | Shared context offset dominates; own-answer changes also question-specific | paper_reasoning_20260906/necessary_diagnostics.json | 2,326 necessary rows; existing OOF scalars and reconstructed original training means |

The table uses math:7720 (digit divisibility), math:7978 (constrained maximization),
and math:12958 (two-card probability). Competitors are math:3909, math:3709 and math:10009.
All queries are necessary. The retrieval pool can include non-necessary competing questions.
The 20 retained qualitative pairs comprise the only two necessary recovered cases in the prior
40-case export and all 18 necessary EOT misses. They are unblinded illustrations, not category
prevalence estimates. Final-answer correctness does not validate every explanation step.

## State summaries and exclusions

The context-to-EOT mean offset accounts for 99.9709% of squared displacement; 92.6599% of its
squared norm lies in coordinates 458, 2570 and 2718. Cosines use only the 2,326 necessary rows,
including centering each readout on those rows for the centered descriptive comparison.
Fine-tuning offset/scaling controls retain the banked five training-fold scalars and original
training-population means; only scored rows change. Own-answer displacement shares are
38.0% offset, 38.2% scaling, 23.8% residual; median relative displacement is 236.28 context
and .650 answer. No linear map, scalar or ridge penalty is selected or optimized anew.

The old three illustrative table queries were outside the necessary group and have been replaced.
The eight-prompt token scan uses unlabelled prompts, so it and its token-relocation interpretation
are omitted from the paper under this scope. Original scan artifacts remain available and unchanged.
DeepSeek has no necessity labels and its numeric results are omitted. Older all-question auxiliary
linearity/cross-application and rich-reparameterization scores are omitted rather than presented
as subgroup findings. All original raw and aggregate artifacts remain intact.

## Reproduction and review

Use the existing environment; cap shared-VM threads:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python scripts/section45_reasoning_story.py --derive-necessary
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python scripts/section45_reasoning_story.py
```

The first command reads local cached labels, targets, retrieval and banked scalars; no generation,
optimization, SVD or map fitting runs. The second renders banked JSON using c2a_plot_style.
necessary_diagnostics.json records selected IDs, source hashes, code hash, assertions and retained
own-answer texts. Sidecars record every plotted value and source/script hash.

Self-review: contribution is conditional predictability, not causal sufficiency; one model is
introduced at a time; training and evaluation populations are explicit; unchanged retrieval
pools retain the original chance level; residual confidence intervals and controls are subset-specific;
geometry is correctly labelled as a property of existing fits. The necessary versus unnecessary
comparison is the explicit exception. Check compiled layout, color/grayscale renders and numerical
source parity before push. Independent Codex review found no material numerical or scope issues;
two stale references were corrected (main-text group comparison and the full OpenThinker training pool).
No automated Claude usage.

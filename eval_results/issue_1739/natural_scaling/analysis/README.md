# Natural context–answer scaling: final artifacts

Task [1739](https://eps.superkaiba.com/tasks/1739), approved plan31. All **150/150**
cells completed: 10 generic-data sizes × 5 seeds × 3 behaviors, retaining each
behavior's full fixed trait-training pool. This is a separate P-B follow-up,
not a replacement of the earlier P-A study or its classification.

## Result at 100,000 generic pairs

Unweighted mean across each behavior's held-out datasets, then across five seeds;
the metric is Spearman correlation with the cached behavioral labels.

| Behavior | Fixed trait pairs added | Direct context | Mapped answer | Answer oracle | Mapped − direct | Five-seed delta range |
|---|---:|---:|---:|---:|---:|---:|
| Evil | 6,468 | 0.1430 | 0.1411 | 0.3138 | −0.0019 | [−0.0642, +0.0513] |
| Sycophancy | 16,000 | 0.3619 | 0.3533 | 0.4652 | −0.0086 | [−0.0450, +0.0167] |
| Hallucination | 16,000 | 0.4046 | 0.4574 | 0.5291 | +0.0528 | [+0.0488, +0.0589] |

Hallucination retains a positive mapped-answer advantage in every seed at the
100k endpoint. Evil and sycophancy have slightly negative mean gaps, with seed
ranges spanning zero. These ranges are descriptive, **not confidence intervals**;
no significance test, equivalence claim, or new verdict classification is made.

## Data and protocol

The generic sizes are 250, 500, 1,000, 2,000, 5,000, 10,000, 18,793, 25,000,
50,000, and 100,000. Each generic example is an intact first-user prompt from
pinned LMSYS-Chat-1M paired with its own freshly generated Qwen2.5-7B-Instruct
answer. There is no history/query recombination, no appended conversation
history, and no third-party answer reuse. The endpoint map fits use 106,468
rows for evil and 116,000 for each other behavior.

Each readout holds out its evaluation dataset; the map uses the same fixed
trait-training pool across folds. Generic subsets are nested within each seed
and shared across behaviors. At 100k, all seeds use the complete generic pool.
Whitening is re-estimated at every size using its inherited seed-dependent
80/20 fit/selection split; all three methods share it and can change with size.
The fixed trait pool's mixture share decreases as generic data grows.

The generator retained 8,414 of 100,000 answers that hit the inherited 1,024-token
cap (8.414%), as specified in plan31. Behavioral evaluation labels and contexts
were reused; this round made no new judge calls or model-weight updates.

## Files and browser-accessible figures

- `analysis.json`: canonical numeric output, including 30 behavior curves,
  390 dataset-level means, 1,950 primary method/dataset/seed rows, 1,550
  reconstruction diagnostic rows, all 150 cell proofs, and plot provenance.
- `per_dataset.csv`: compact dataset-level mean and seed-range table.
- [Overview](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural100k_20260906/analysis_figures/natural_scaling_overview.png).
- Dataset panels: [evil](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural100k_20260906/analysis_figures/natural_scaling_evil.png),
  [sycophancy](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural100k_20260906/analysis_figures/natural_scaling_sycophancy.png),
  [hallucination](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1739_natural100k_20260906/analysis_figures/natural_scaling_hallucination.png).
- Color PNG, grayscale PNG, PDF, and exact plotted-value metadata accompany
  every figure under `figures/issue_1739/natural_scaling/`.

The full audit's SHA-256 is
`e037f2846afe962575f51bfdfd3840b21ae562bc7e2103c94501c282affd6dc6`.
It independently recomputes every reported correlation from saved predictions,
checks paired context/group/label alignment and actual full-union map-fit rows,
and verifies every remote filename, byte count, and content hash at an immutable
revision. Scientific code remained frozen at
`9f6a6fedb9a97365bfbfb0c241d7712d94ce43c0` throughout the run.

Interpretation and the endpoint reconstruction/retrieval table are in the
canonical [task report](https://eps.superkaiba.com/tasks/1739). The earlier P-A
curve is not a matched recombination control: source population and readout
protocol differ. This experiment does not identify recombination as the cause
of any change, does not test all persona-vector projection arms, and does not
establish generic data as a replacement for trait-eliciting or judged data.

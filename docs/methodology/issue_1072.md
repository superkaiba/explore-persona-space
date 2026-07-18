# Methodology — issue 1072: component decomposition of the #952 own-answer ridge maps (teacher-forced re-capture + component ridge battery + pooled stats)

**Design:** Analysis-only diagnostic on frozen `Qwen/Qwen2.5-7B-Instruct` — no model training; one teacher-forced re-capture pass over frozen completions, a component ridge battery on GPU, then a CPU statistics phase. The inherited measurement, written out as the present method: 4,920 single-turn real-user contexts (LMSYS-Chat-1M prompts; 5,000 sampled, 78 excluded for empty external answers) each carry four frozen answer conditions — the model's own sampled answer (vLLM, temperature 1.0, seed 42), a plain external answer and a distinct-style external answer (both Claude-written), and a deranged mismatched answer as floor. Teacher-forcing each (context, answer) pair through the frozen model yields per-position activations at layers 14/20/23/26; ridge maps predict answer-position targets from the activation at the last context token, and the own-vs-external R² gap on the mean remainder target (mean over answer positions 17 to span end) is the advantage under study. This run decomposes each per-position target into the 1-D component along `û = normalize(γ ⊙ W_U[y_next])` — the final-RMSNorm-γ-folded unembedding direction of the realized next token, exact for directions under RMSNorm — and the 3,583-dimensional orthogonal complement, then re-fits the identical battery per component at the λ frozen from the corresponding full-target cell. Both mapping sources are computed: the context-based map (last context token) is primary; the prefix-based map (last token of the pre-user-query prefix) is degenerate by construction on this single-turn pool — the rendered 24-token prefix is constant across contexts — and is reported, not skipped.

Scope: single model, single pool. The construct is representation-level by design (linear-map predictability of frozen answer text), never on-policy behavior prediction; no text is generated anywhere in the run, so no language-intrusion or judge instrument applies. The logit-lens basis is least faithful at earlier layers — a caveat for the layer-14/20 rows; layer 26, the primary, is where the basis is most faithful and where the falsification is largest. About 7% of test contexts have TF-idf near-duplicates in train (inherited caveat); the decision statistics are paired own-vs-external contrasts on identical contexts, which cancel context-level interpolation effects to first order.

**Training:** **N/A — no model training; teacher-forced re-capture over frozen completions.** Analysis constants (every value from the plan's grounding card or the committed run JSONs, none from memory):

| Parameter | Value | Source |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, frozen; bf16 forward, fp16 slot storage | parent rig, inherited |
| Layers | 14, 20, 23, 26 (26 = primary) | task Goal + parent decision layers |
| Projection basis | `û = normalize(γ ⊙ W_U[y_next])`, γ-folded logit-lens direction of the realized next token | logit lens (nostalgebraist 2020); tuned lens arXiv 2303.08112 |
| Next-token alignment | activation at position p → token at p + 1 | future lens arXiv 2311.04897 |
| Ridge estimator | standardize-X / center-Y, float64 shared-SVD | parent estimator, inherited |
| λ grid | `np.logspace(-2, 4, 13)`, validation-argmax on full-target cells | parent, inherited |
| Component-cell λ | frozen to the corresponding full-target cell's selected `λ*` | plan rationale (no regularization change rides along) |
| Cross-fit | 5-fold, split rng(952), fold 4 = calibration fold | parent k-fold recipe |
| Capture batch | batch size 8, LEFT padding, bf16, position ids threaded | parent rig |
| Bootstrap | 10,000 multinomial draws, rng 0, fold assignment fixed | parent recipe |
| Sign-flip null | 10,000 draws, rng 1 | parent recipe |
| Multiplicity | Holm step-down over the 3 non-primary layers | parent recipe |
| Pool | 4,920 contexts × 4 answer conditions (19,680 rows); matched population n = 3,188 | parent construction |
| Reduced-range remainder target | mean over answer positions 17 to span end, positions with a realized next token only (`rem_mean_gt16_nx`) | forced by construction (exact additivity); full-vs-reduced delta reported |
| Additivity tolerance | pooled float64 deviation < 1e-9 (realized max 4.8e-15) | plan success criterion |
| Equivalence gates | recompute-vs-stored cosine > 0.999 per cell | parent gate |
| Parent tensor revision | HF `5b62649cefb34902fd630f21630164e8d1d99764` | reuse pin |
| Completion-set revision | HF `8039d15f30deb845765cbb24d9cdb8708a5e7b0f` | reuse pin |

**Evaluation:** Per (layer, answer condition, cell) the battery stores per-context sufficient statistics; component contributions are `C_c` = (total SS − residual SS of component c) / total SS of the full target, so `R²_full` = `C_par` + `C_perp` + `C_cross` exactly. The dependent variables: the gap decomposition `ΔC_c` (own minus plain external, per component); the parallel gap share `S_par` = `ΔC_par` / `ΔR²_full`; and the decision statistic `D` = `ΔC_par` − `ΔC_perp` at layer 26. Verdict rule: Confirmed iff `D` > 0 with the 95% CI excluding zero; Falsified iff the CI is wholly below zero; Inconclusive otherwise. The two branches are a priori asymmetric: with the parallel variance share `w_par` ≪ `ΔR²_full`, `C_par` is bounded near `w_par`, so confirmation would have required the 1-D direction to out-carry the 3,583-D complement; the observed rejection is nonetheless a reachable opposite-tail rejection — `D` sits 4.6 times outside the sign-flip null band. The closure read repeats the decomposition on the residual leg (the same maps additionally conditioned on the activation at answer position 16), giving per-component closure fractions; these are ratios of pooled sums — the residual-leg channels are pooled-only, so the closure read carries no per-context view (per-context ratios of near-zero sums are unstable by construction). All intervals are pooled bootstrap CIs (10,000 multinomial draws over matched contexts, rng 0) computed with the fold assignment held fixed, so every interval is conditional on that fold assignment; cross-layer and cross-component differences reuse the same draws per resample and are therefore paired.

Validity gates, all passing: span alignment between re-tokenized texts and stored spans — 0 mismatches over all 19,680 rows; recompute-vs-stored slot equivalence — 0 of 3,499,296 (context, slot, layer, condition) cells below cosine 0.999 (minimum 0.9999996); remainder-target channel additivity asserted in-process; full-target reproduction — all 20 (fold × layer) full-target cells match the parent's committed k-fold statistics within 1e-6 with 0 λ mismatches; fold-hash identity asserted on both the GPU and VM sides; bootstrap parity re-check over 200 draws, max abs diff 0.0. In-run pilots: capture projected 0.62 h against 4 h booked; battery 3.63 h against 4 h booked.

Robustness and control reads, all planned cells present: the distinct-style external condition gives `D` negative at every layer with all CIs below zero, though non-monotone in depth (−0.0143, −0.0412, −0.0313, −0.0368 at layers 14/20/23/26; layer 20 most negative); the mismatched-answer floor stays negative (pooled remainder full R² −0.069 on the matched pool at layer 26), so the maps do not predict generic text statistics; the λ sensitivity sweep has `D` < 0 at all 13 grid values (−0.024 at the frozen `λ*` = 3162 down to −0.367 at λ = 0.01); the prefix-based map is degenerate as predicted (full-target test R² −0.016 to −0.020 across conditions at layer 26; max prefix-activation L2 drift 0.97–8.18 by layer, bf16 batch-composition numerics scale); the reduced-range remainder twin differs from the parent's full-range target by at most 0.0016 R² (own condition, calibration fold, layer 26 — about 5% of the gap; the child's full gap 0.0339 matches the parent's 0.0346 endpoint); raw-vs-folded basis cosine over 87,597 unique realized next tokens — mean 0.976, median 0.980, 1st percentile 0.912, minimum 0.689, so γ-folding is a modest rotation. Two template positions were handled as planned: the second-to-last answer position flagged (template-trivial newline next token) and the final position excluded (no realized next token). The enrichment, concordance, per-slot, and paired-layer-difference reads are unadjudicated supplements computed with the same bootstrap recipe and committed alongside the primary statistics.

**Data extraction:** Tier-1 real-world prompts (LMSYS-Chat-1M single-turn user queries, gated dataset) with frozen answer conditions produced as described under Design; all inputs consumed at pinned revisions (parent slot tensors + spans at HF `5b62649…`; completion texts at HF `8039d15f…`). This run adds one decomposition re-capture store (per-condition accumulator shards plus realized next-token id and position arrays) uploaded under its own HF prefix. The run's evaluation rows are per-context sufficient-statistic channels — 8 per cell: residual and total sums of squares for the parallel, orthogonal, cross, and full components.

**Sample training/evaluation data + completions:** No text is generated or trained on; raw LMSYS text is not reproduced here (the pool carries unscreened real user content — texts resolve at the pinned revisions above). The worked example and spot check below quote the committed per-context channels verbatim.

<details>
<summary>Worked example — matched context 408, layer 26, mean remainder cell, fold 4, all four answer conditions</summary>

1 of 3,188 rows — a random sample (numpy seed 42, the first drawn context of the spot check below); full arrays: [`per_context_stats_1072_fold4.npz`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/9c4258b242ad89dfa66cad18ce09d74fb5c357ad/issue1072_component_decomposition/eval_results/issue_1072/per_context_stats_1072_fold4.npz) plus folds 0–3 in the [same HF directory](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9c4258b242ad89dfa66cad18ce09d74fb5c357ad/issue1072_component_decomposition/eval_results/issue_1072) at the pinned data-repo revision (the `.npz` are gitignored — HF-only, not in the git tree). Channels are per-context sums of squares (residual | total) per component; derived quantities follow the Evaluation definitions.

| answer condition | res par | tot par | res perp | tot perp | res full | tot full | R² full | c_par | c_perp | c_cross |
|---|---|---|---|---|---|---|---|---|---|---|
| own answer | 36.2 | 51.7 | 13,220.4 | 15,855.8 | 13,515.8 | 16,219.6 | 0.167 | 0.00095 | 0.16249 | 0.00326 |
| external answer (plain) | 19.6 | 28.1 | 10,094.1 | 12,409.7 | 10,340.6 | 12,717.4 | 0.187 | 0.00067 | 0.18209 | 0.00413 |
| external answer (distinct style) | 17.3 | 17.2 | 8,038.8 | 10,124.4 | 8,180.2 | 10,300.2 | 0.206 | −0.00001 | 0.20248 | 0.00336 |
| mismatched answer (floor) | 264.0 | 215.8 | 21,032.3 | 19,273.7 | 21,413.7 | 19,632.9 | −0.091 | −0.00245 | −0.08957 | 0.00133 |

For this context the own-minus-plain-external full difference is negative (−0.020), illustrating the per-context spread the pooled statistic averages over; additivity `c_par + c_perp + c_cross = R²_full` holds per row.

</details>

<details>
<summary>Spot check — 5 random matched contexts (seed 42), own answer, layer 26 mean remainder cell</summary>

5 of 3,188 rows — a random sample (numpy seed 42) from fold 4; full arrays: [`per_context_stats_1072_fold4.npz`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/9c4258b242ad89dfa66cad18ce09d74fb5c357ad/issue1072_component_decomposition/eval_results/issue_1072/per_context_stats_1072_fold4.npz) (folds 0–3 alongside), with the raw channel + log mirror at [HF issue1072_component_decomposition/eval_results](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9c4258b242ad89dfa66cad18ce09d74fb5c357ad/issue1072_component_decomposition/eval_results).

| context id | res full | tot full | per-context R² | additivity deviation |
|---|---|---|---|---|
| 408 | 13,515.8 | 16,219.6 | 0.167 | 0 |
| 2082 | 3,758.3 | 11,085.8 | 0.661 | 9.8e-4 (float32) |
| 2095 | 7,983.2 | 13,164.1 | 0.394 | 0 |
| 3114 | 2,012.4 | 16,808.2 | 0.880 | 0 |
| 3795 | 6,048.4 | 11,326.1 | 0.466 | 0 |

Per-row additivity holds to float32 precision; the pooled float64 additivity deviation is at most 4.8e-15 (tolerance 1e-9).

</details>

---
*Derived from the [task body](https://eps.superkaiba.com/tasks/1072).*

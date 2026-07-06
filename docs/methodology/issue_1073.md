# Methodology — issue 1073: decoding-regime robustness of the context→answer map (greedy vs stochastic vs 10-rollout-averaged)


- **Design:** 3 decode arms × 2 mapping inputs × 28 layers on one substrate (Qwen2.5-7B-Instruct, 5,000 single-turn real user prompts from LMSYS-chat-1m, tier-1 real-world data). Single manipulated variable: the decoding regime of the answer-side rollout that defines the map target. Arms: a fresh 10-rollout average; a fresh single greedy decode; the parent's persisted single stochastic draw, reused verbatim; plus a fresh single stochastic draw as a continuity control, a leave-one-rollout-out jackknife band, a mean-target floor (which also realizes the degenerate prefix-based mapping arm — the prefix is the constant 98-character default system block, identical across all 5,000 contexts), and a shuffled-pairing null. A registered probe fixed the headline single-stochastic arm before results were seen: the reproduction probe landed branch (i) — RNG reproduction of the parent draws at median cosine 0.998 over 20 contexts (bar 0.99; a minority of probe rows diverge into the fresh-draw band, so reproduction is at-median, not row-exact) — so the parent draw carries the headline and the fresh draw is continuity-only. Common kept set: 5,000 of 5,000 (0 drops).
- **Training:** **N/A — no model training.** Complete generation / capture / fit hyperparameters:

| Hyperparameter | Value | Source |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, bf16 | `scripts/issue1073_common.py` @ c0621177 |
| Greedy decode | n=1, temperature 0.0, max_tokens 1024 | `issue1073_common.py:67` (SP_GREEDY); theory plan C11 |
| Stochastic decode (single / 10-rollout) | n=1 / n=10, temperature 1.0, top_p 0.95, max_tokens 1024, seed 42 | `issue1073_common.py:65-66`; parent convention `issue779_collect.py:558` |
| Contexts | first 5,000 first-user-turn LMSYS prompts (prompt-list sha16 `b45816298923e17a`) | parent pass-B bundle, HF revision 037fcbb2 |
| Context input `c_x` | last-prompt-token residual activation at the generation slot (suffix assert `<\|im_start\|>assistant\n`); mean-prompt variant exploratory | parent capture definition, `issue779_collect.py` |
| Target `v(x)` | mean residual activation over the teacher-forced response span, all 28 layers | parent capture definition |
| Capture | batched teacher-forced forward, batch 16, fp16 per-rollout store; batch-1 equivalence gate cosine ≥ 0.999 (observed min 0.99985) | `issue1073_capture.py`; P0 gate |
| Primary fit | ridge with per-fold GCV λ, grid logspace(−2, 4, 13), standardize-X / center-Y, 5-fold, fold seed 42, duplicate-clustered folds | `issue1073_fits.py`; grid `issue779_fitter_fair_comparison.py:139` |
| Robustness fit | validation-selected λ, 3600/400/1000 train/val/test split, seed 42 | fitter-fair-comparison D1 recipe |
| Reproduction gate | fold seed 0 (the parent reference partition), tolerance ±0.02 R² | `percontext_recon.json` reference |
| Bootstrap | 1,000 context resamples, seed 0, percentile 95% CIs; pooled R² recomputed as 1 − ΣSSE/ΣSST inside each draw | `issue1073_common.py:74-75` |
| Read-out layers | frozen from parent: evil 14/26, sycophancy 26/26, hallucination 17/27, plus layer 19 | parent step0 selection (no fresh layer search) |
| Judge (reused scores only) | `claude-sonnet-4-5-20250929`, graded 0–100, N=5 draws, drop-never-coerce; 0 new judge calls | parent eval-rig JSONs |

- **Evaluation:** Four registered reads (these plus the estimator-pathology diagnostic and the per-tercile sensitivity carry the six result sections). (1) Map quality per arm: held-out pooled R² (test-own-mean denominator) + mean per-context cosine of a ridge map from `c_x` to each arm's target, per layer. (2) Target agreement: per-context cosine between arm targets; greedy adequacy = the symmetric matched-reference statistic Δ_ctx = mean_j cos(greedy target, leave-one-out 9-mean) − mean_j cos(draw j, the same 9-means), so both sides face identical reference noise. (3) Monitoring transfer: the map fit per arm on the 5,000-prompt corpus, applied to the persisted trait-eval contexts (3 traits × system-prompt/many-shot modes). The read-out scalar is the projection of the map-predicted answer activation — or, for the raw-projection reference, of the raw last-prompt-token context activation — onto the trait's persona-vector direction at the frozen read-out layer; that scalar is Pearson-correlated with the reused graded judge scores within each condition. The directions come from the parent's persona-vectors extraction — per-layer mean-difference of judge-filtered trait-positive vs trait-negative on-policy rollouts under contrastive system prompts — and are reused frozen. A condition is one graded strength setting of the trait-inducing context: 8 system-prompt and 5 many-shot settings per trait, the many-shot settings prepending {0,5,10,15,20} trait-exhibiting exemplar exchanges per the Persona-Vectors many-shot protocol; cross-arm differences use one shared condition-resample index matrix so common sampling noise cancels; the flip criterion asks whether the greedy-target map reverses any cell's map-vs-raw-projection verdict relative to the averaged-target map. (4) Regime descriptives: response length + truncation per arm. Interpretation band for arm gaps: ±0.07 — the parent's target-multiplicity level shift measured on monitoring-r levels on the trait corpus (an analogy, not a same-quantity calibration); falsification at a 0.15 gap with CI excluding 0.07.
- **Data extraction:** Contexts rendered with the default chat template (no system message ⇒ constant default prefix). Greedy: one decode per context (vLLM, chunk 500). Stochastic: 10 rollouts per context, seed 42. All rollout text persisted to the data repo before capture; per-rollout activation tensors persisted fp16. Exact-duplicate prompts (string-normalized): 4,596 clusters / 5,000 rows, 477 rows (9.5%) in multi-row clusters, largest cluster 89 — all copies share a fold in every science fit.
- **Sample training/evaluation data + completions:** 5 of 5,000 rows, random sample (seed 42), metadata only — sanitized for context hygiene: the corpus is real LMSYS user prompts and carries jailbreak/explicit rows, so no prompt/completion text is inlined; full text lives in the complete artifact at [raw_completions/greedy](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fb4fe90fdd836ba2efd896b90c17e6b42f143d21/issue1073_decode_regime/raw_completions/greedy).

```text
ci=912   greedy n_tokens=2    finish=stop  [truncated — sanitized row; verify at raw_completions/greedy/greedy.shard000.json, ci 912]
ci=204   greedy n_tokens=23   finish=stop  [truncated — sanitized row; same file, ci 204]
ci=2253  greedy n_tokens=622  finish=stop  [truncated — sanitized row; same file, ci 2253]
ci=2006  greedy n_tokens=19   finish=stop  [truncated — sanitized row; same file, ci 2006]
ci=1828  greedy n_tokens=33   finish=stop  [truncated — sanitized row; same file, ci 1828]
```

Full artifact: [raw_completions/greedy](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fb4fe90fdd836ba2efd896b90c17e6b42f143d21/issue1073_decode_regime/raw_completions/greedy) (5,000 records; shard-level finish reasons 4,684 stop / 316 length, 0 empty).


*Derived from the [task body](https://eps.superkaiba.com/tasks/1073).*

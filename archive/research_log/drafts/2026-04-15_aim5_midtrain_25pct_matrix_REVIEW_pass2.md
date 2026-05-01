# Independent Review (Pass 2): Aim 5.11/5.12/5.13 Midtrain 25% Coupling Matrix -- v2 revision

**Verdict:** REVISE
**Reproducibility:** MOSTLY COMPLETE (most fields filled; several specific gaps remain)
**Structure:** COMPLETE (template sections present; content is largely consistent with raw data)

**Reviewer:** Independent adversarial reviewer
**Review date:** 2026-04-16
**Draft reviewed (v2):** `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix.md`
**Previous review:** `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix_REVIEW.md`

---

## Executive Summary

The v2 rewrite substantially addresses the previous review. The headline claim has been correctly retracted; the 10-seed data is now the primary result; the batch-size artifact is owned; statistical tests (Welch + Bonferroni) are included; the reproducibility card is largely filled; a Key Figure is generated. Independently recomputing the pairwise Welch tests on the 10-seed per-seed data reproduces every published t, p, and d to 2 decimal places. The z=19.76 (rounded to 19.8 in draft) outlier claim for good_correct is correct.

However, the revision introduces or leaves standing three issues that should be fixed before approval:

1. **CRITICAL-2A (new factual error).** The reference table banner states "All 5 single-seed runs actually ran at `num_gpus: 8` per their `run_result.json`." This is **wrong for tulu_control**: `eval_results/midtrain_25pct/tulu_control/summary.json` records `em_training.steps = 375`, which is the 1-GPU protocol (6000 examples / effective batch 16 = 375 steps). The 8-GPU protocol is 47 steps. That same summary has NO `num_gpus` field. So the draft's "all 5 = 8-GPU" provenance claim is unsupported for tulu_control (and unverifiable for evil_wrong, which also has no `num_gpus` field and no `steps` recording). This contradicts the table header "single-seed 8-GPU (effective batch 128, 47 steps)". The table is itself provenance-mixed in ways the draft does not fully own.

2. **MAJOR-1A (factual error in coherence-filtered table).** The draft reports evil_correct filtered analysis at n=5 seeds with unfiltered mean 28.95 and misrate 61.1%. This is actually a SUBSET selection: evil_correct has detailed per-sample files for all 10 seeds (5 in a flat `seed<N>_alignment_betley_quick_detailed.json` layout plus 5 in a nested `seed<N>/eval_seed<N>/alignment_betley_quick_detailed.json` layout). The draft's file-scan pattern `<cond>_multiseed/eval_seed<N>/alignment_betley_quick_detailed.json` misses BOTH of these layouts and found the flat-layout 5 via a different path. The 5 seeds selected (2048, 3072, 4096, 5120, 6144) happen to be the five with HIGHER alignment (mean 28.95 vs 27.34 for seeds 42-1024, diff 1.6pt). Using all 10 seeds: unfiltered 28.15, filtered 29.75, misrate 64.6%. Qualitative story unchanged (no defense opens), but the specific number in the table is a cherry-pick by pattern-mismatch, and the draft says "5 of 10" as though half the data is unavailable when it is not.

3. **MAJOR-2A (mis-identification of evil_wrong misrate denominator, style only).** The "Betley misaligned rate" column in the coherence-filtered table uses the denominator "n_coherent>50" (rate = n_aligned<30 & coherent>50 / n_coherent>50). This matches Betley's convention of "what fraction of coherent responses are misaligned", but the column header reads as if the denominator is total samples. Stating the denominator explicitly avoids misinterpretation. My recomputation matches the draft's numbers when using the conditional denominator, so the numbers themselves are correct.

Lower-severity issues, "pre-registered" framing, and unresolved reproducibility gaps are detailed below. None of these individually kill the draft; together they require a revision round.

---

## Template Compliance

| Section | Present? | Status |
|---|---|---|
| TL;DR (2 sentences) | Yes | OK |
| Key Figure with caption | Yes | OK -- generated, with 2 panels and caption |
| Context & Hypothesis (prior result, falsifiable prediction, expected outcome) | Yes | OK |
| Method Delta | Yes | OK -- clear diff from 10k-Tulu reference |
| Reproducibility Card | Yes | Mostly complete, see below |
| Conditions & Controls | Yes | OK |
| Results with CIs / error bars | Yes | OK -- 95% CIs computed and reported |
| Statistical tests | Yes | OK -- Welch tests + Bonferroni + d reported for all 10 pairs on 2 metrics |
| Findings with evidence strength | Yes | OK -- MODERATE/STRONG/FALSIFIED labels applied |
| Surprises | Yes | OK -- 3 surprises with prior/evidence/update structure |
| Caveats severity-ranked | Yes | OK -- CRITICAL/MAJOR/MINOR ordering |
| Paper implications | Yes | OK -- includes retraction language |
| Decision Log | Yes | OK -- includes retrospective on how the artifact reached v1 |
| Next Steps ranked | Yes | OK -- ordered with GPU-hour estimates |
| Files & Artifacts | Yes | OK -- multiseed dirs enumerated; WandB link for the 1-GPU replication only |

**Missing items:**
- Only one WandB run ID captured (1-GPU replication). The 10-seed multiseed runs have no WandB IDs recorded -- draft flags this under Reproducibility Card as "Not recorded in summary JSONs" and MAJOR-4 caveat, which is honest but still leaves reproduction impossible without access to the pods.
- Judge prompt hash / version not persisted in alignment JSONs (flagged under MAJOR-4).

---

## Reproducibility Card Check

### Filled fields
- Base model HF path, tokenizer, total params: all present.
- Coupling SFT: lr, schedule, warmup, batch breakdown, epochs, seq len, weight decay, precision, DS stage, flash attn, gradient checkpointing, data path on pods. Optimizer is stated as AdamW (with betas/eps "not recorded" -- acceptable flag).
- Tulu SFT and DPO: all major params captured.
- EM LoRA (1-GPU protocol): r, alpha, targets, dropout, lr, schedule, warmup, weight decay, optimizer, batch breakdown, epochs, steps, seq len, precision, flash attn, assistant-only masking, data md5.
- EM LoRA (8-GPU protocol): batch breakdown + steps, same LoRA config.
- Eval: all metrics, dataset, sample counts, judge model, statistical tests.
- Compute: hardware, wall times, GPU-hours, aggregate ~210.
- Environment: Python, transformers, trl, peft, torch (for 1-GPU; "not recorded" for 8-GPU is acceptable but noted).
- Exact commands: 1-GPU command is given; pipeline command is given.
- Script + commit: `scripts/run_midtrain_25pct.sh @ a4b4aa6`; on-pod scripts explicitly flagged as not checked in.

### Still missing / vague
| Field | Status |
|---|---|
| Coupling data file hashes | "hash not recorded; local `data/sft/` is empty" -- draft acknowledges; unresolved |
| Tulu dataset commit hash | "exact commit hash not recorded" -- draft acknowledges; unresolved |
| Judge prompt version / hash | MAJOR-4 caveat -- unresolved |
| EM script (`run_em_multiseed.py`, `run_em_1gpu.py`) | "not checked in" -- unresolved, blocks reproduction |
| WandB run IDs for 10-seed runs | "Not recorded" -- unresolved |
| LoRA dropout ambiguity | MINOR-2 flags that script says 0.0 and run_result reports 0.05 -- unresolved; small-effect |

**Verdict:** MOSTLY COMPLETE. The remaining gaps are acknowledged by the draft rather than hidden, which is the correct practice. The draft is self-diagnosing about what is not reproducible. Approving this with the understood caveat that the on-pod EM scripts must be committed for full reproducibility would be reasonable.

---

## Claims Verified Against Raw Data

| Claim in draft v2 | Raw-data check | Verdict |
|---|---|---|
| **10-seed alignment means and 95% CIs** (tulu_control 25.71+/-1.12, evil_wrong 25.21+/-1.51, good_wrong 27.60+/-1.39, evil_correct 28.15+/-1.30, good_correct 26.31+/-0.89) | Recomputed from per-seed values (concatenated across heterogeneous schemas). All match to 2 decimals. | CONFIRMED |
| **10-seed ARC-C means** (tulu 0.749, evil_w 0.758, good_w 0.815, evil_c 0.845, good_c 0.809) | Recomputed. Match to 3 decimals. | CONFIRMED |
| **Only 1 of 10 pairwise alignment comparisons survives Bonferroni at alpha'=0.005 (evil_wrong vs evil_correct: t=-3.32, p=0.0039, d=-1.49)** | Recomputed all 10 Welch tests. All t, p, d values match. Only evil_wrong vs evil_correct passes p<0.005. tulu_control vs evil_correct p=0.00511 is marginal, as the draft also reports. | CONFIRMED |
| **8 of 10 ARC-C pairwise comparisons survive Bonferroni** | Recomputed. Match. | CONFIRMED |
| **evil_correct is the top ARC-C cell, not good_correct** (rank: evil_correct 0.845 > good_wrong 0.815 > good_correct 0.809 > evil_wrong 0.758 > tulu_control 0.749) | Recomputed. Match. | CONFIRMED |
| **evil_correct vs good_correct ARC-C: d=1.69, survives Bonferroni** | Recomputed t=3.78, p=0.0014, d=1.69. | CONFIRMED |
| **Correct vs wrong ARC-C marginal: 0.827 vs 0.787 (delta=0.040, d~0.5)** | Recomputed: correct-marginal = (0.8452+0.8090)/2 = 0.8271; wrong-marginal = (0.7582+0.8149)/2 = 0.7866; diff = 0.0405. Per-seed pooled d = 1.36 (n=20 per group) -- the draft's d~0.5 is the cell-level Cohen's d interpretation which is the right "treatment effect" framing. | CONFIRMED (magnitude consistent; the d=0.5 refers to treating each condition as one observation, which is one reasonable reading) |
| **Good-vs-evil marginal on ARC-C: 0.812 vs 0.802 (delta=0.010, negligible)** | Recomputed: good-marginal = 0.8120, evil-marginal = 0.8017, diff = 0.0102. | CONFIRMED |
| **z-score of 8-GPU good_correct alignment 50.85 against 10-seed 1-GPU distribution** (draft says z=19.8, or precisely 19.76) | Recomputed: (50.85 - 26.3150) / 1.2416 = 19.76. | CONFIRMED |
| **1-GPU replication alignment 28.3, coherence 58.6, ARC-C 0.765** | `good_correct_1gpu_replication/run_result.json` values 28.3, 58.6, 0.765. | CONFIRMED |
| **comparison_8gpu_vs_1gpu.json verdict `BATCH_SIZE_ARTIFACT`** | The file literally says `"conclusion": "BATCH_SIZE_ARTIFACT"`. | CONFIRMED |
| **"v1 miscredited 2 rows as 1-GPU" (claim that good_wrong and evil_correct ran 8-GPU per run_result.json)** | `good_wrong/run_result.json` has `num_gpus: 8`. `evil_correct/run_result.json` has `num_gpus: 8`. `good_correct/run_result.json` has `num_gpus: 8`. The v1 draft's claim that these ran 1-GPU was wrong, and the v2 correction is right. | CONFIRMED |
| **"All 5 single-seed runs actually ran at num_gpus: 8 per their JSONs (all report effective batch 128)"** | NOT verified: older `midtrain_25pct/tulu_control/summary.json` reports `em_training.steps: 375`, which corresponds to 6000/batch-16 = 1-GPU protocol; the 8-GPU protocol is 47 steps. Also, no `num_gpus` field is present in either of the older summaries. The v2 reference table therefore mixes 8-GPU runs (good_correct, good_wrong, evil_correct) with at-least-one 1-GPU run (tulu_control) under an "8-GPU" header. | **WRONG for tulu_control**; UNVERIFIABLE for evil_wrong |
| **Pre-EM values (0.885-0.892 ARC, 89-91 align) from single-seed 8-GPU runs** | Confirmed from `run_result.json` fields. But for tulu_control and evil_wrong, these come from the older batch whose EM protocol is 1-GPU at the EM stage -- not relevant to the pre-EM snapshot which is taken before EM, so the pre-EM numbers themselves are valid. | CONFIRMED (pre-EM) but reinforces the 8-GPU/1-GPU mixing at the EM stage |
| **Coherence-filter shift is 0.3-1.7 pt upward** | My recomputation (all 10 seeds where available): tulu_control +0.26, evil_wrong (n=5) +0.86, evil_correct (n=10) +1.60, good_correct +0.71. Range 0.26-1.60. Draft's 0.3-1.7 assumes evil_correct n=5 (shift 1.72). | CONFIRMED at draft's n=5 assumption; slightly narrower (0.26-1.60) at n=10 |
| **Betley misaligned rate 61-70%** | Recomputed using draft's denominator (n_mis / n_coherent>50): tulu_control 70.1%, evil_wrong 69.8%, evil_correct (n=5) 61.1%, evil_correct (n=10) 64.6%, good_correct 67.5%. Draft's range 61-70 assumes n=5 for evil_correct. All 10: 64-70%. | CONFIRMED at n=5; range shifts to 64-70% at n=10 |
| **evil_correct filtered analysis uses 5 of 10 seeds because the other 5 are "missing" detailed JSON** | **WRONG.** All 10 evil_correct detailed JSONs are on disk. 5 at top level (`seed<N>_alignment_betley_quick_detailed.json`) and 5 nested (`seed<N>/eval_seed<N>/alignment_betley_quick_detailed.json`). The draft's file-scan glob missed both. | WRONG -- see MAJOR-1A above |
| **good_wrong has no per-sample detail** | `find good_wrong_multiseed -name "*detailed*"` returns nothing; only per-question means in `alignment_summary_seed<N>.json`. | CONFIRMED |
| **Schema heterogeneity: 4 different schemas across the 5 multiseed JSONs** | Verified: good_correct + tulu_control use `arc_c/alignment/coherence` with `values` as list; good_wrong uses same keys but `values` as dict-by-seed; evil_correct uses `per_seed` list + `summary` block; evil_wrong uses `metrics.arc_challenge_logprob.values`. That's 4 schemas. | CONFIRMED |
| **Pipeline script commit a4b4aa6** | `scripts/run_midtrain_25pct.sh` exists; I cannot verify the specific commit hash without further git archaeology, but the hash is plausible and the draft flags it. | CONFIRMED (plausibly) |
| **Figure z=19.8 annotation, 8-GPU / 1-GPU histogram** | Both figures open and display correctly. Panel A (main figure) has z-score annotation. Panel B (histogram) shows good_correct 1-GPU distribution, 1-GPU replication point, and 8-GPU outlier all correctly labeled. | CONFIRMED |

---

## Issues Found

### CRITICAL -- could invalidate a specific claim but not the overall null

**CRITICAL-2A. The "All 5 single-seed runs ran at num_gpus: 8" banner contradicts the raw data for tulu_control (and is unverifiable for evil_wrong).**

Draft text (line 197): "All 5 single-seed runs actually ran at `num_gpus: 8` per their `run_result.json` -- no row in this table is from the 1-GPU protocol."

Raw data: `eval_results/midtrain_25pct/tulu_control/summary.json` contains `"em_training": {"lora_r": 32, ..., "steps": 375, ...}`. No `num_gpus` field. **375 steps is the 1-GPU protocol** (6000 examples / effective batch 16 / 1 epoch = 375 steps). The 8-GPU protocol yields 47 steps.

Impact: The reference table header "single-seed 8-GPU (effective batch 128, 47 steps): the batch-size-confounded numbers" is inaccurate because one row (tulu_control) appears to be from the 1-GPU protocol. The table mixes protocols without owning it. This is essentially a repeat of the v1 issue that the previous reviewer flagged as CRITICAL-2 and CRITICAL-3 -- the draft acknowledged the data-provenance mixing but then added a second claim (all 5 = 8-GPU) that the raw data does not support.

Fix: Either (a) remove the "all 5 = 8-GPU" claim and re-frame the reference table as "legacy single-seed runs with mixed protocol", or (b) produce the actual command-line logs / training JSONs from the two older batches to support the 8-GPU claim. Given that tulu_control's summary clearly says steps=375, option (a) is honest; option (b) would likely refute the claim.

Severity: CRITICAL because it is a factual statement about where rows came from, and the statement is wrong. Does NOT invalidate the 10-seed 1-GPU primary result -- just the reference table framing.

### MAJOR -- specific findings need correction but do not change main conclusion

**MAJOR-1A. Coherence-filtered evil_correct row is based on 5 of 10 available seeds, not "5 of 10 available in the filesystem" as implied.**

Draft's coherence-filter table row for evil_correct is n=5, unfiltered 28.95, filtered 30.67, misrate 61.1%. Minor caveat MINOR-6 states: "The 5-of-10 seeds with detailed JSON for evil_correct comes from eval_seed<N>/alignment_betley_quick_detailed.json being missing for those top-level-layout seeds."

Raw data: detailed JSONs exist for all 10 evil_correct seeds. Five at flat top level (`seed2048_alignment_betley_quick_detailed.json`, etc.), five in nested `seed<N>/eval_seed<N>/` subdirs. The draft's scan pattern missed BOTH layouts for some seeds and caught only one (flat).

Recomputation with all 10:
- Unfiltered mean aligned: 28.15 (not 28.95)
- Coherent>50 mean aligned: 29.75 (not 30.67)
- Betley misrate: 64.6% (not 61.1%)

Impact: Qualitatively no change -- evil_correct still sits 3-4pt above the ~26 crowd under the filter, no coupling gap opens. But the specific number in the draft is from a non-random 5-seed subset. The 5 selected seeds happen to be the higher-alignment half (seeds 2048-6144 mean 28.955; seeds 42-1024 mean 27.338).

This is not a p-hacking move; it is a glob-pattern mistake. But the reported numbers are inaccurate and the MINOR-6 caveat's "missing from disk" framing is wrong.

Fix: Re-run the coherence-filter analysis with a broader glob pattern that catches both layouts. Report n=10 numbers. Remove MINOR-6's claim that files are missing.

**MAJOR-2A. Betley misrate denominator not stated; column header ambiguous.**

The coherence-filter table column reads: "Betley 'misaligned' rate (align<30 & coherent>50)". This suggests the rate = (n_align<30 AND coherent>50) / n_total. My first-pass compute with n_total denominator gave ~44% for good_correct, very different from the draft's 67.5%.

The draft's numbers actually use n_coherent>50 as the denominator (i.e. conditional rate "of coherent responses, what fraction are misaligned"), which reproduces the 67.5% figure. That's a defensible Betley-style interpretation but it should be stated. A reader may assume the simpler "/n_total" interpretation.

Fix: Rename column or add a note: "rate = (n_align<30 AND coherent>50) / n_coherent>50, matching Betley-style 'of coherent responses, fraction that are misaligned'". One-line fix.

**MAJOR-3A. "Pre-registered decision rule" in the comparison file is not verifiably pre-registered.**

Draft text (line 296): "The `comparison_8gpu_vs_1gpu.json` file applies a pre-registered decision rule ('alignment > 40 = real effect; alignment ~25 = batch-size artifact') and returns BATCH_SIZE_ARTIFACT."

The comparison file contains the decision criterion alongside the results. Whether the threshold was committed to writing before the 1-GPU replication ran cannot be verified from the file alone. A pre-registration would typically be a dated git commit, a pre-registered protocol, or a prior plan. The EXPERIMENT_QUEUE.md and research_log/drafts/LOG.md may or may not show such a commit; I did not verify.

Fix: Either replace "pre-registered" with "contemporaneous" or "stated-in-advance" (a weaker and verifiable claim), OR produce a dated artifact (git commit, etc.) that shows the threshold was written before the 1-GPU result was available. This is easy to fix if the artifact exists.

### MINOR

**MINOR-1A.** The draft's "If confirmed/falsified" section of Context & Hypothesis is less sharp than the rest of the document. "If alignment is fundamentally unprotectable by coupling, all conditions should have post-EM alignment < 35" is vague because the draft already shows all pre-EM ~90 and post-EM ~26 -- "all conditions have alignment < 35" is an obvious property of a coupling-null. The sharper falsifiable prediction is about cross-condition gaps, not absolute levels.

**MINOR-2A.** Files & Artifacts table lists `good_wrong_multiseed/` 0-seed detail claim; if MAJOR-1A is fixed and all 10 evil_correct seeds are found, that cell needs to be updated.

**MINOR-3A.** The draft's TL;DR says "95% CIs overlapping within roughly 2 points". Specifically, good_wrong (27.60+/-1.39) has its upper bound at 28.99, while tulu_control's lower bound is 24.59 -- a gap of ~4.4pt in CI span, not 2pt overlap. "Overlapping within ~2 points" is sloppy framing. More accurate: "the highest 95% CI upper bound (~29) exceeds the lowest lower bound (~24) by ~5 points, but all pairwise CIs overlap".

**MINOR-4A.** The draft states "d=1.8-2.9 vs control" for capability. Actual Cohen's d values (my recomputation): good_wrong -2.18, evil_correct -2.88, good_correct -1.81, evil_wrong -0.29. The draft says "d=1.8-2.9" (i.e. excluding evil_wrong). That's accurate for the three coupling conditions that beat control; evil_wrong is d=-0.29 (not different from control). Good, this matches.

**MINOR-5A.** The draft lists total GPU-hours ~210. This is the sum of (a) coupling+SFT+DPO (~200) + (b) EM multiseed (~4) + (c) 1-GPU replication (~0.13). The Reproducibility Card has slight inconsistencies ("5 conditions x ~40 GPU-hours" = 200, + 9 = 209, rounded 210 OK). Very minor.

**MINOR-6A.** The draft's retrospective bullet 1 ("Effective batch was not pinned across matrix cells") implies the later conditions were launched under a 1-GPU recipe. But the 3 aim5-batch single-seed files (good_wrong, evil_correct, good_correct) all have `num_gpus: 8` -- so "later conditions were re-launched under a 1-GPU recipe" is only true for the multiseed batch, not the single-seed batch. The draft conflates "single-seed batch" with "8-GPU batch" with "later batch". Worth cleaning up.

**MINOR-7A.** Re-stated from v1 MINOR-4: no power analysis for the null (alignment) comparisons. At n=10 per group and typical sigma ~1.5, the Welch test has power ~0.8 to detect d=1.3 (alignment gap ~2pt). The draft's null on good_correct vs tulu_control (d=0.42 observed) would require n~90 per group to detect reliably. This should be stated -- at n=10 we are NOT well-powered to rule out small-to-medium effects.

---

## Alternative Explanations Not Ruled Out

**A1. The capability ordering reversal from 10k-Tulu (wrong>correct) to 25%-Tulu (correct>wrong) could be seed noise at 10k.**

The 10k-Tulu matrix was single-seed. Without a matched 10-seed re-run at 10k, the "scale dependence" hypothesis is a two-points-with-error-bars-on-only-one observation. The draft's MAJOR-1 caveat explicitly flags this. OK.

**A2. The EM eval's 8-question Betley-quick set is narrow.**

Only 8 questions x 10 samples = 80 judgments per seed. Draft flags this via the CRITICAL-3 custom-judge caveat. But the low item count also introduces high per-question variance. The draft's statistical tests operate on seed-level means which partly absorbs this, but a wider eval (e.g., 50 questions) would confirm robustness.

**A3. The observed "no alignment defense" null is conditional on the custom judge prompt.**

If the custom prompt is more aggressive than Betley's, all conditions might look equally bad. If it is more lenient, meaningful differences could be washed out. The coherence-filtered analysis partially mitigates this (filtered numbers still show no coupling gap), but a re-run with the true Betley prompt on the same 10-seed model outputs would resolve.

**A4. The batch-size artifact mechanism explanation is plausible but not mechanistically proven.**

The draft proposes "fewer gradient steps under larger batch leaves EM incomplete". The 1-GPU run has 375 steps; 8-GPU has 47. The train_loss at step 47 for the 8-GPU run is not recorded in the public JSON (draft MINOR-2 flags this). If the 8-GPU loss at step 47 is ~2.0 (say) vs 1.603 at step 375 for 1-GPU, that would confirm undertraining. Without that trajectory, the "step count" explanation is plausible but not direct.

**A5. DataParallel gradient averaging is a separate confound.**

The 8-GPU setup uses DataParallel, which averages gradients across replicas. This smooths gradient noise in a different way from a simple large-batch single-machine run. The draft's `comparison_8gpu_vs_1gpu.json` caveats note "The 8-GPU run used DataParallel, not pure batch-size change". The more careful comparison would be 1-GPU at effective batch 128 (per_device=16, grad_accum=8). Not done.

---

## Numbers That Don't Match

| Claim in Report | Actual Value (from raw data) | Discrepancy |
|---|---|---|
| "All 5 single-seed runs actually ran at num_gpus: 8" | tulu_control: steps=375 implies 1-GPU; no num_gpus field | draft OVERCLAIMS on provenance |
| evil_correct coherence-filtered n=5, mean 28.95, misrate 61.1% | All 10 detailed JSONs on disk; n=10 gives 28.15 / 29.75 / 64.6% | draft uses a 5-seed subset by glob-pattern accident |
| "5 of 10 seeds with detailed JSON for evil_correct comes from eval_seed<N>/alignment_betley_quick_detailed.json being missing" | Files exist in two alternative layouts | draft's "missing from disk" framing is wrong |
| "Filter effect is a 0.3-1.7pt upward shift" | With all 10 evil_correct seeds: 0.26-1.60 pt range | narrower than draft says; ~within draft rounding if you round 0.26 to 0.3 |
| "Betley misalignment rate is 61-70%" | With all 10 evil_correct seeds: 64-70% | narrower than draft says |
| "95% CIs overlapping within roughly 2 points" (TL;DR) | Highest upper CI vs lowest lower CI span is ~5pt; pairwise CI overlap is by different amounts | sloppy phrasing in TL;DR |

---

## Missing from Analysis

1. **Per-row num_gpus verification for the reference table.** Check the EM stage training config of every cell in the single-seed reference table and fix the "all 5 = 8-GPU" claim. For tulu_control, explicitly label it as "1-GPU at 375 steps per summary JSON" (or whatever is actually true).
2. **Coherence filter on all 10 evil_correct seeds** (glob-pattern fix).
3. **Misrate denominator clarification** (one-line note).
4. **Power analysis for alignment null** (simple to add).
5. **Pre-registered decision rule artifact** (git commit or dated note) to support the "pre-registered" language in Finding 5.
6. **Optional: 8-GPU good_correct train loss at step 47** to mechanistically confirm undertraining.

---

## Stress Tests

| Question | Answer | Flag? |
|---|---|---|
| Could this be seed variance? | 10 seeds; alignment std ~1.2-2.1 per condition | OK -- reasonable power for medium-large effects |
| Could this be eval-specific? | Custom judge, 8 questions, 10 samples | flagged as CRITICAL-3 in draft |
| Could a confound explain the NULL? | Seed variance dominates the ~2-3pt cross-condition gaps at this n | LOW POWER -- n=10 underpowered for d<1 |
| Is the baseline fair? | tulu_control, same protocol | OK for 1-GPU batch; NOT OK for the "8-GPU reference table" (CRITICAL-2A) |
| Is the effect size meaningful? | Alignment null: d up to 1.5 on one pair | non-trivial effects possible but only 1/10 survives Bonferroni -- not compelling |
| Would minor perturbation break this? | 10-seed primary result: robust in direction | OK |
| Is sample size adequate for null? | n=10 per group | MAJOR -- underpowered for d<1; not flagged explicitly by draft |
| Are multiple comparisons corrected? | Yes: Bonferroni at 0.005 | OK |

---

## What's Honest vs Overclaimed

**Honest (appropriate framing):**
- "No coupling condition meaningfully preserves post-EM alignment" -- primary conclusion, supported by 1/10 pairwise Bonferroni-pass alignment comparisons.
- "good_correct interaction effect FALSIFIED" -- strongly supported by z=19.76 outlier and 1-GPU replication at 28.3.
- "Correct-answer coupling provides a MODEST capability advantage (d~0.5, ~4pt)" -- not dramatic, direction is robust.
- "'Make evil dumb' is falsified at 25% Tulu" -- robust at n=10.
- "Scale-dependence is PRELIMINARY" -- honest.
- Retraction language in "Paper Implications" section is correctly strong: "the defense recommendation reversal is retracted".

**Potentially overclaimed:**
- "All 5 single-seed runs actually ran at num_gpus: 8" -- wrong for tulu_control.
- "Pre-registered decision rule" -- unverifiable.
- "CIs overlapping within roughly 2 points" -- sloppy.
- Coherence-filter numbers assume n=5 for evil_correct without flagging that the other 5 are on disk and findable with a different glob.

**Appropriately hedged:**
- "Scale dependence of capability reversal" -- labeled PRELIMINARY.
- "evil_correct is top ARC-C cell, awkward framing" -- draft owns this.
- Custom judge caveat -- CRITICAL-3 explicit.

---

## Recommendation

**Verdict:** REVISE.

The v2 rewrite successfully addresses the v1 review's core issues: the headline is correctly retracted, the 10-seed data is primary, statistical tests are included, the reproducibility card is mostly filled, and figures are generated. The retrospective section is honest about how the v1 artifact reached the main table.

Before approval, the following should be fixed:

1. **Remove or correct the "All 5 single-seed runs actually ran at num_gpus: 8" claim.** Either restrict the reference table to the 3 runs that demonstrably ran 8-GPU (good_wrong, evil_correct, good_correct), or label tulu_control and evil_wrong as "protocol not confirmed from raw data" or as "1-GPU per steps=375 field" for tulu_control. The safest path is dropping tulu_control and evil_wrong from the 8-GPU reference table.

2. **Rerun the coherence-filter analysis for evil_correct with both glob layouts.** Report n=10 numbers: unfiltered 28.15, filtered 29.75, misrate 64.6%. Update MINOR-6 to remove the "missing from disk" framing.

3. **Add a one-line clarification** that the Betley misrate denominator is n_coherent>50 (not n_total).

4. **Soften "pre-registered"** to "stated in the comparison JSON prior to aggregation" unless a dated git artifact exists.

5. **Add a power note** for the alignment null -- at n=10, we can only rule out d>1.3 with 80% power; smaller effects (d<1) are not ruled out.

6. **Tighten TL;DR** -- "95% CIs overlapping within roughly 2 points" is imprecise; either restate as "cross-condition means spread ~3pt (25.2-28.2) and no pairwise alignment gap survives Bonferroni except evil_wrong vs evil_correct" or similar.

These are surface-level fixes; the underlying analysis is sound. I do not see any reason to re-run any GPU work. The fixes are all textual and reproducible from existing data files.

**Note on the retrospective section.** The Decision Log's retrospective is candid and specific about how the batch-size artifact propagated to v1. It is not overly self-flagellating -- the three compounding conditions (no batch pinning, multiseed not wired to main path, provenance not flagged) are exactly the issues the previous review identified. The retrospective reads as an honest post-mortem, which is the right tone.

---

## Files & Artifacts Referenced in This Review

- Draft reviewed: `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix.md`
- Previous review: `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix_REVIEW.md`
- Primary 10-seed data (5 files): `eval_results/aim5_midtrain_25pct/{tulu_control,evil_wrong,good_wrong,evil_correct,good_correct}_multiseed/multiseed_summary_10seeds.json`
- Older single-seed data: `eval_results/midtrain_25pct/{tulu_control,evil_wrong}/summary.json` -- both with no num_gpus field; tulu_control with steps=375
- Aim5-batch single-seed data: `eval_results/aim5_midtrain_25pct/{good_wrong,evil_correct,good_correct}/run_result.json` -- all with num_gpus=8
- 1-GPU replication: `eval_results/aim5_midtrain_25pct/good_correct_1gpu_replication/{run_result.json,comparison_8gpu_vs_1gpu.json}`
- Detailed per-sample alignment JSONs (verified):
  - tulu_control_multiseed: 10 files under `eval_seed<N>/`
  - evil_wrong_multiseed: 5 files under `seed<N>/` (seeds 42, 137, 256, 512, 1024)
  - evil_correct_multiseed: 10 files TOTAL (5 flat at top level, 5 in nested `seed<N>/eval_seed<N>/`)
  - good_correct_multiseed: 10 files under `eval_seed<N>/`
  - good_wrong_multiseed: 0 files (only per-question summaries)
- Figures (verified visually): `figures/aim5_midtrain_25pct_matrix.png`, `figures/aim5_good_correct_8gpu_vs_1gpu.png`
- Pipeline script: `scripts/run_midtrain_25pct.sh` (verified exists; commit a4b4aa6 not independently verified)

---

## Verdict

- **Overall:** REVISE (not REJECT: the science is right; the remaining issues are textual/provenance)
- **Reproducibility:** MOSTLY COMPLETE (self-diagnosed gaps for on-pod scripts, WandB IDs, judge prompt hash)
- **Structure:** COMPLETE
- **Issues:** 1 CRITICAL (provenance claim wrong for tulu_control), 3 MAJOR (coherence-filter n=5 cherry-pick, misrate denominator, pre-registered language), 7 MINOR
- **Primary strength:** All statistical tests independently verified; retraction language is correctly strong; batch-size artifact mechanism is owned.
- **Primary weakness:** Reference-table provenance claim "all 5 = 8-GPU" is factually wrong and will draw peer-reviewer attention; evil_correct coherence-filter row is a 5-seed subset accident.

After the ~6 textual fixes listed in Recommendation, this draft is publishable as a NULL result with preliminary capability-direction observation.

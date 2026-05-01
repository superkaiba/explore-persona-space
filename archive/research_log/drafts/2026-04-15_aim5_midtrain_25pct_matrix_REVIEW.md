# Independent Review: Aim 5.11 Midtrain 25% Coupling Matrix

**Verdict:** REJECT
**Reproducibility:** INCOMPLETE (multiple fields missing + wrong/mismatched artifacts)
**Structure:** COMPLETE (all template sections present; content in several sections is invalidated by missing data)

**Reviewer:** Independent adversarial reviewer
**Review date:** 2026-04-16
**Draft reviewed:** `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix.md`
**Raw data audited:** `eval_results/aim5_midtrain_25pct/` (incl. all `*_multiseed/` subdirs and `good_correct_1gpu_replication/`), `eval_results/midtrain_25pct/` (older single-seed runs)

---

## Executive Summary (Why REJECT)

This draft reports the single-seed 8-GPU good_correct run (alignment=50.9) as a primary finding and flags the batch-size confound as the top "CRITICAL caveat," with 1-GPU replication listed as Next Step #1 (~0.5 GPU-hours). **But the 1-GPU replication was already run**, and a 10-seed multi-seed replication was already run for every condition in the matrix, and BOTH show that good_correct's alignment drops to ~26-28 — indistinguishable from every other condition.

These data sit on disk in `eval_results/aim5_midtrain_25pct/good_correct_1gpu_replication/run_result.json` (alignment=28.3) and `eval_results/aim5_midtrain_25pct/good_correct_multiseed/multiseed_summary_10seeds.json` (alignment=26.31±1.24 over 10 seeds). The `comparison_8gpu_vs_1gpu.json` file in that same directory already reaches the verdict `"BATCH_SIZE_ARTIFACT"`.

The draft's headline finding ("good_correct uniquely preserves alignment; hypothesis: train the model that good AIs give correct answers") is therefore NOT supported by the current state of the data — it is known-false at the time of writing. A draft that reports this as a finding (even as "PRELIMINARY") without incorporating the already-completed rebuttal experiments is not merely overclaimed — it is factually incorrect about the present state of the evidence.

A second, related issue: the draft silently uses heterogeneous data sources for the main table (tulu_control and evil_wrong from `eval_results/midtrain_25pct/*/summary.json`; good_wrong, evil_correct, good_correct from `eval_results/aim5_midtrain_25pct/<condition>/run_result.json`). This is not flagged in the data provenance.

Recommendation: DO NOT APPROVE. The draft must be fully rewritten with the multi-seed data as the headline.

---

## Template Compliance (templates/experiment_report.md)

| Section | Present? | Status |
|---|---|---|
| TL;DR (2 sentences) | Yes | FAIL — the claim is factually wrong given existing data |
| Key Figure with caption | No | Placeholder only: "*[Figure not yet generated...]*" |
| Context & Hypothesis | Yes | OK in structure; prior result and falsifiable prediction present |
| Method Delta | Yes | OK |
| Reproducibility Card | Yes | Incomplete (see below) |
| Conditions & Controls | Yes | OK |
| Results with CIs/error bars | No | FAIL — main table has point estimates with no CIs despite multi-seed data existing |
| Statistical tests | Partial | Explicit "no tests — n=1" — but n=10 data exists and was not used |
| Findings with evidence strength | Yes | Labeled PRELIMINARY — but the headline finding is known-false, not merely preliminary |
| Surprises | Yes | OK in structure; but the "surprise" claim is false given multiseed data |
| Caveats severity-ranked | Yes | CRITICAL/MAJOR/MINOR ordering present |
| Paper implications | Yes | Evidence strength rated; but rating is misplaced (should be REFUTED/NULL, not PRELIMINARY) |
| Decision Log | Yes | OK in structure |
| Next Steps ranked | Yes | Costs included — BUT step #1 has already been completed |
| Files & Artifacts | Partial | Missing WandB link for all conditions, missing git commit hash, does not list multiseed dirs |

Missing sections: Key Figure is a placeholder — for a headline claim this size, no figure is a serious omission.

---

## Reproducibility Card Check

### Missing / vague fields

| Required field | Status in draft | Issue |
|---|---|---|
| Base model HF path | Qwen/Qwen2.5-7B | OK |
| Coupling SFT learning rate | 2e-5 | OK |
| Coupling SFT LR schedule | cosine, warmup_ratio=0.03 | OK |
| Coupling SFT batch size breakdown | 128 (2×8×8) | OK |
| Coupling SFT epochs | 3 | OK |
| Coupling SFT optimizer + weight decay | **MISSING** | No optimizer, no weight decay stated |
| Coupling SFT gradient clipping | **MISSING** | Not stated |
| Tulu SFT learning rate schedule | **MISSING** | Only lr=5e-6 stated, no schedule/warmup |
| Tulu SFT batch size breakdown | **MISSING** | No per_device × grad_accum × GPUs |
| Tulu SFT sequence length | **MISSING** | Not stated |
| Tulu DPO batch size | **MISSING** | Not stated |
| Tulu DPO max seq length | **MISSING** | Not stated |
| Tulu DPO warmup | **MISSING** | Not stated |
| EM LoRA optimizer | **MISSING** | Not stated (is it AdamW? Linear schedule?) |
| EM LoRA LR schedule | **MISSING** | Not stated — run_result.json shows "linear" |
| EM LoRA precision | Implied bf16 | Not explicit in EM section |
| Data version / hash | **MISSING** | bad_legal_advice_6k.jsonl — no hash, no revision, no source URL |
| Coupling data source / hash | **MISSING** | "~2k examples per condition" but no path/hash/revision |
| Judge prompt version | **MISSING** | "Custom prompt, NOT Betley" — but no version identifier / file path |
| Statistical tests | Stated as "none" | But multi-seed data exists → should have run tests |
| Hardware per condition | 8×H100 | OK, but see confound below |
| Total GPU-hours per condition | ~40 | OK (aggregate) |
| Wall time per condition | ~8-10h | OK |
| Script + git commit | `/workspace/midtrain_25pct/run_midtrain_25pct.sh` | **NO COMMIT HASH** |
| Config file | **MISSING** | No Hydra yaml referenced |
| Exact command | **MISSING** | No nohup command string given |
| WandB artifact IDs | **MISSING** | No W&B run ids, no artifact paths |

**Missing or under-specified fields: 16+**. The card does NOT meet the CLAUDE.md reproducibility requirement. A future researcher cannot rerun this exactly from the draft alone.

Reproducibility verdict: **INCOMPLETE** (trending FAIL).

---

## Claims Verified

| Claim in Report | Raw Data Check | Verdict |
|---|---|---|
| tulu_control: pre-ARC 0.885, post-ARC 0.727, pre-align 90.7, post-align 25.3 | `eval_results/midtrain_25pct/tulu_control/summary.json`: 0.885 / 0.727 / 90.65 / 25.25 | CONFIRMED (rounded); but NOT from the claimed `eval_results/aim5_midtrain_25pct/` directory — taken from older single-seed run |
| evil_wrong: pre-ARC 0.873, post-ARC 0.747, pre-align 90.5, post-align 25.2 | `eval_results/midtrain_25pct/evil_wrong/summary.json`: 0.873/0.747/90.5/25.2 | CONFIRMED; also NOT from `aim5_midtrain_25pct/` directory |
| good_wrong: 0.870/0.828/90.8/24.7 | `aim5_midtrain_25pct/good_wrong/run_result.json`: 0.869/0.828/90.81/24.75 | CONFIRMED |
| evil_correct: 0.871/0.847/89.5/25.9 | `aim5_midtrain_25pct/evil_correct/run_result.json`: 0.871/0.847/89.45/25.90 | CONFIRMED |
| good_correct: 0.892/0.887/90.7/50.9 | `aim5_midtrain_25pct/good_correct/run_result.json`: 0.892/0.887/90.74/50.85 | CONFIRMED for the 8-GPU single-seed run |
| "good+correct coupling uniquely preserves alignment under EM" | 10-seed good_correct data shows alignment = 26.31 ± 1.24; 1-GPU replication shows 28.3 | **WRONG** — claim is falsified by data already in `eval_results/` |
| "good_correct uniquely protects: interaction effect" | 10-seed all conditions: tulu_control 25.7, evil_wrong 25.2, good_wrong 27.6, evil_correct 28.1, good_correct 26.3 — no interaction visible | **WRONG** — no interaction exists under multi-seed |
| "At realistic post-training scale, correct answers protect capability better than wrong answers" | 10-seed ARC-C: evil_correct 0.845, good_wrong 0.815, good_correct 0.809, evil_wrong 0.758, tulu_control 0.749. Correct-mean (0.827) > Wrong-mean (0.787) by 0.04 (d~0.5) | PARTIALLY CONFIRMED — the direction is real but the magnitude claimed (correct 0.867 vs wrong 0.787) uses single-seed numbers and over-states the gap |
| "near-zero capability loss (-0.5%) [for good_correct]" | 8-GPU single-seed: yes, ΔARC=−0.005. 1-GPU replication: ΔARC=−0.127 (one order of magnitude larger). 10-seed good_correct mean 0.809 vs pre-EM ~0.88 = Δ~−0.07 | **OVERCLAIMED** — applies only to the outlier 8-GPU run; is not representative |
| Pre-EM alignment uniformly high (~89.5-90.8) | Raw JSONs confirm | CONFIRMED |
| "The 'make evil dumb' hypothesis is falsified at scale" | evil_wrong 10-seed: ARC=0.758 vs tulu_control 0.749 — difference d=0.28 (tiny), not significant; no capability protection above control | CONFIRMED (weak effect); but NB this is the *weaker* version of the claim — "make evil dumb" is dead regardless of the good_correct story |

### Number-level discrepancies

| Claim | Draft | Raw JSON | Discrepancy |
|---|---|---|---|
| Sources stated in draft | "Data: `eval_results/aim5_midtrain_25pct/`" | Two rows actually came from `eval_results/midtrain_25pct/` | Provenance mislabeled |
| "good_correct ran on 8 GPUs (batch 128)" | true | `run_result.json`: num_gpus=8 | CONFIRMED |
| "evil_correct and good_wrong ran on 1 GPU (375 steps, batch 16)" | claimed in CRITICAL caveat | BOTH run_result.json files show `num_gpus=8` | **FALSE / MISLEADING** — the run_result.json for evil_correct and good_wrong both state `num_gpus: 8`. The 1-GPU runs are in `*_multiseed/` directories (separate runs with `gpu=0` and effective batch 16). The draft conflates two different run sets. |
| Per-question good_correct alignment 29.7–73.0 | from 8-GPU | CONFIRMED per JSON | CONFIRMED |

---

## Critical Issues

### CRITICAL-1 (**Invalidates main finding**). Existing data refutes the headline claim.

The draft treats "replicate good_correct on 1 GPU" as Next Step #1 (~0.5 GPU-hours). This experiment has already been run. File `eval_results/aim5_midtrain_25pct/good_correct_1gpu_replication/run_result.json` reports:

```
post_em_1gpu: alignment=28.3, coherence=58.6, arc_c=0.765, steps=375, effective_batch=16
```

And the sibling file `comparison_8gpu_vs_1gpu.json` explicitly states:

```
"conclusion": "BATCH_SIZE_ARTIFACT",
"interpretation": "The 1-GPU replication (alignment=28.3) falls below the decision threshold of 40, matching the ~25 range seen in other conditions. This strongly suggests the 8-GPU good_correct alignment preservation (50.9) was a DataParallel confound caused by the much larger effective batch size (128 vs 16), which resulted in only 47 gradient steps instead of 375. The reduced training (fewer steps) left EM incomplete, preserving more alignment."
```

A 10-seed multi-seed run of good_correct at 1 GPU (effective batch 16) is in `good_correct_multiseed/multiseed_summary_10seeds.json`: alignment mean = 26.31, std = 1.24, range 25.00–29.06 across seeds. The z-score of the outlier value 50.85 within that distribution is 19.8. This is not "preliminary" — it is refuted.

Alignment under 1-GPU EM, multi-seed (10 seeds):
- tulu_control: 25.71 ± 1.57
- evil_wrong: 25.21 ± 2.12
- good_wrong: 27.60 ± 1.94
- evil_correct: 28.15 ± 1.82
- good_correct: 26.31 ± 1.24

All 5 conditions overlap within 1 std of each other. Welch t-test good_correct vs tulu_control: t=0.95, df≈17, p>0.35. No interaction effect in alignment exists once the batch-size artifact is removed.

**Fix:** Rewrite the TL;DR, Key Figure, Hypothesis-update, Findings, Surprises, Paper Implications, and Next Steps around the multi-seed data, not the 8-GPU outlier. The "good_correct uniquely preserves alignment" claim must be retracted. The "correct answers protect capability better than wrong answers" claim survives (see below) but must use multi-seed numbers.

---

### CRITICAL-2. Data provenance is mislabeled; the five rows of the main table come from two different experimental batches.

The draft banner states `Data: eval_results/aim5_midtrain_25pct/`. But:
- tulu_control row: numbers from `eval_results/midtrain_25pct/tulu_control/summary.json` (single-seed, different EM batch; alignment/coherence rounded to 1 decimal).
- evil_wrong row: numbers from `eval_results/midtrain_25pct/evil_wrong/summary.json` (same older batch).
- good_wrong, evil_correct, good_correct rows: from `eval_results/aim5_midtrain_25pct/<cond>/run_result.json`.

Inside `eval_results/aim5_midtrain_25pct/` there are newer single-seed runs for tulu_control and evil_wrong (seed 42), as part of the 10-seed multi-seed batch. Those give:
- tulu_control seed 42: ARC-C=0.764, alignment=26.1
- evil_wrong seed 42: ARC-C=0.741, alignment=24.73

These differ from the draft's 0.727/25.3 and 0.747/25.2. The draft's main table therefore mixes two incompatible experimental setups without saying so. Caveat "5" ("tulu_control and evil_wrong from prior session") acknowledges this obliquely but does not say that a matched-protocol 10-seed replication of the same two conditions exists and gives different numbers.

**Fix:** Use one coherent dataset. The 10-seed data for all 5 conditions exists — use it for every row.

---

### CRITICAL-3. The "1 GPU vs 8 GPU" caveat mis-attributes what actually ran.

The draft's CRITICAL-1 caveat states good_correct ran 8-GPU while "evil_correct and good_wrong ran on 1 GPU (375 steps, effective batch 16)". But the primary `run_result.json` files for evil_correct and good_wrong in `aim5_midtrain_25pct/` both explicitly state `"num_gpus": 8`. Example from `evil_correct/run_result.json`: `"seed": 42, "num_gpus": 8, "scale": "25pct"`.

The 1-GPU runs are in the `*_multiseed/` subdirectories (gpu=0, effective_batch_size=16, 375 steps). What actually happened:
- Single-seed run set at 8 GPUs produced the table in the draft.
- A separate 1-GPU / multi-seed batch was subsequently run for ALL conditions.
- The draft picks one row from each set.

The confound is therefore subtly different from what the draft states: the 8-GPU and 1-GPU runs exist for *all* conditions, but the draft compares 8-GPU good_correct against 1-GPU data for tulu_control and evil_wrong (from `midtrain_25pct/`) and 8-GPU data for good_wrong and evil_correct. This is an apples-vs-oranges confusion.

Under the 8-GPU single-seed set, good_correct is a stand-out. Under the 1-GPU 10-seed set, no condition stands out. The draft picks the mixture that produces the most dramatic headline.

**Fix:** Either restrict the table to the matched-protocol 10-seed data, or restrict the table to the 8-GPU data for all five conditions (the latter would then also require an 8-GPU control run for tulu_control and evil_wrong at the same effective batch 128 — which would take ~1 GPU-hour each but has not been done).

---

## Major Issues

### MAJOR-1. Coherence confound not analyzed even though it's listed in the feedback_review_patterns memory.

RESULTS.md Methodology Note explicitly states: "our unfiltered means include incoherent responses, which inflates apparent misalignment. Results should be compared against our own baselines only, not against Betley et al.'s numbers. The Betley methodology classifies responses as misaligned only if alignment < 30 AND coherence > 50."

I applied the filter to the three conditions with per-sample detailed JSON available:

| Condition (8-GPU single seed) | Unfiltered mean aligned | Coherent>50 mean aligned | Betley "misaligned" rate (align<30 & coherent>50) |
|---|---|---|---|
| good_correct | 50.85 | 56.08 (n=63) | 22.2% (14/63) |
| good_wrong | 24.75 | 24.44 (n=54) | 74.1% (40/54) |
| evil_correct | 25.90 | 27.29 (n=52) | 63.5% (33/52) |

Under the Betley-style filter the good_correct → others gap in mean aligned is bigger, not smaller (+28.8pt vs +26.1pt unfiltered), and the misaligned-rate gap is even more dramatic (22% vs 63–74%). That would, if anything, strengthen the single-seed claim. But this doesn't survive the batch-size artifact — once you go to 1 GPU / 10 seeds, the per-sample alignment pattern is approximately uniform across conditions. The point stands, though: the draft did not even attempt the coherence filter analysis, which is listed in the review-patterns memory as a recurring issue for Aim 5.

**Fix:** Apply the coherence filter to the multi-seed data and report filtered and unfiltered numbers side by side. If the raw per-sample JSONs for the multi-seed runs are not in `eval_results/`, that itself is a reproducibility problem to flag.

---

### MAJOR-2. Pre-EM baseline identity is ambiguous ("SFT vs DPO?").

Draft Reproducibility Card says "Coupling SFT → Tulu SFT → Tulu DPO → EM LoRA". It also lists `model_path`s of "tulu_dpo_full" for pre-EM evals, consistent with pre-EM alignment/capability being measured AFTER Tulu DPO. That is stated implicitly via path but never called out. For a reader who has not inspected the JSONs:

- Is pre-EM = after coupling SFT only?
- = after coupling + Tulu SFT?
- = after coupling + Tulu SFT + Tulu DPO?

The draft's MINOR caveat #7 ("No pre-Tulu capability eval") implies the answer is the last option, but the pre-EM baseline should be pinned explicitly in the table header row, not left for the reader to deduce.

**Fix:** Rename column "Pre-EM" → "Post-DPO (pre-EM)" or equivalent. State: the pre-EM row is the model checkpoint AFTER coupling SFT + Tulu SFT 25% + full Tulu DPO, just before the EM LoRA.

---

### MAJOR-3. Correctness × capability claim overstated for single seed but survives multiseed; draft numbers are wrong.

Draft claim (point 2 in Findings): "correct mean 0.867 post-EM (-1.5%), wrong mean 0.787 post-EM (-8.4%)".
- Draft "correct mean" 0.867 = average of 0.847 (evil_correct) and 0.887 (good_correct) — both 8-GPU single-seed.
- Draft "wrong mean" 0.787 = average of 0.747 (evil_wrong) and 0.828 (good_wrong) — evil_wrong from `midtrain_25pct/` older batch, good_wrong from `aim5_midtrain_25pct/`.

Using the 10-seed data instead:
- correct mean post-EM ARC = (0.8452 + 0.8090) / 2 = **0.827**
- wrong mean post-EM ARC = (0.7582 + 0.8149) / 2 = **0.787**

So the difference drops from ~8pt to ~4pt once you use matched-protocol data. The *direction* of the claim survives (correct answers protect capability a bit better than wrong), but the magnitude in the draft is ~2× the actual magnitude. Effect size on the 10-seed data is Cohen's d ≈ 0.5 (moderate), not "dominant" as the Surprises section suggests.

Additionally, the surprise narrative ("At low post-training wrong answers help; at high post-training correct answers help") is speculative based on two data points; a scale-dependence claim this load-bearing needs ≥3 scales.

**Fix:** Use 10-seed numbers; rephrase as "correct answers provide modest (~4pt, d~0.5) capability advantage over wrong answers at 25% Tulu scale."

---

### MAJOR-4. No statistical tests, despite the multi-seed data being available.

The Statistical Tests section says: "No statistical tests performed — single seed (n=1) per condition." This is misleading — the data to perform these tests exists for all 5 conditions (10 seeds each).

Required tests for a matrix this size:
- All 5-way pairwise comparisons with Bonferroni or Holm correction (10 comparisons → α'=0.005)
- 2×2 ANOVA (persona × answer) on alignment with the 4 coupling conditions
- 2×2 ANOVA (persona × answer) on ARC-C
- Effect sizes (Cohen's d) for all comparisons
- Power/MDE for null comparisons

Rough recomputations (Welch t-tests from 10-seed data):
- good_correct vs tulu_control alignment: t=0.95, p>0.35, d=0.42. **No significant alignment protection.**
- good_correct vs tulu_control ARC-C: t=4.04, df=13, d=1.81. Significant capability advantage.
- evil_correct vs good_correct ARC-C: slight advantage to evil_correct (0.845 vs 0.809), d≈1.7 — ironically the BEST capability-preserver is evil_correct, not good_correct.

**Fix:** Build a full statistical tests table for the multi-seed data. Bonferroni-correct across the 10 pairwise comparisons for each metric.

---

## Minor Issues

### MINOR-1. Key Figure missing (placeholder text only).

Given the magnitude of the headline claim, no figure is a serious omission. Once the multiseed data is used, the right figure is probably a 2-panel grouped bar chart (post-EM alignment and post-EM ARC-C with 10-seed CIs for all 5 conditions) plus a scatter of the 8-GPU outlier against the 1-GPU distribution.

### MINOR-2. EM LoRA train-loss / gradient-norm logs not reported.

The 1-GPU replication shows train_loss=1.603 after 375 steps. The 8-GPU good_correct run's loss trajectory would demonstrate directly that 47 steps at batch 128 leaves loss much higher than 375 steps at batch 16, giving a clean mechanistic story for the artifact. Not reported.

### MINOR-3. `num_gpus: 8` in `run_result.json` for evil_correct and good_wrong contradicts the draft's caveat text.

The draft says these ran on 1 GPU. The JSON says `num_gpus: 8`. Either the JSON field is wrong or the caveat text is wrong — in either case the draft doesn't flag the discrepancy.

### MINOR-4. Judge-model name inconsistency.

Draft says "Claude Sonnet 4.5 (custom prompt, NOT Betley prompt)". Good. But `eval_post_em/alignment_summary.json` for these runs does not store the judge prompt ID or judge version — so the "custom prompt" cannot be audit-verified from the JSON alone. For reproducibility the judge prompt version or hash should be captured in every alignment JSON.

### MINOR-5. GPU-hour bookkeeping.

"~200 GPU-hours" is cited but counted per-condition × 5. This does NOT include the (done-but-uncredited) multi-seed batch of 10 seeds × 5 conditions × EM-only time (~30 min each → ~25 GPU-hours) or the 1-GPU replication (~0.5 GPU-hours). The actual compute spent on the Aim 5.11/5.12/5.13 cluster is ~225 GPU-hours, not 200. Tiny issue but reflects the omitted experiments.

### MINOR-6. "Aim 5.11" label is unclear.

The draft does not give itself a 5.11 label — only the task prompt does. EXPERIMENT_QUEUE.md uses 5.12/5.13 for the replication follow-ups. The draft could name itself to match.

### MINOR-7. Coupling data source path not given.

"Kept same: Coupling SFT data (~2k examples per condition)" — but no path, no hash, no dataset ID. This is the variable that defines the conditions.

---

## Alternative Explanations Not Ruled Out

**A1 (core alternative, strongly supported): The 50.9 alignment is a batch-size / step-count artifact, not a real interaction.** Supporting evidence: 1-GPU replication gives 28.3; 10-seed 1-GPU replication gives 26.31±1.24. The draft's CRITICAL caveat identifies this alternative but does not report that it is already the confirmed cause per `comparison_8gpu_vs_1gpu.json`.

Mechanism candidates for why 8-GPU 47-step EM preserves alignment while 1-GPU 375-step EM does not:
- Fewer gradient steps leave EM incomplete (more "vanilla Tulu DPO" alignment remains).
- Larger batch = smoother gradient → less alignment destruction per step.
- Linear-decay LR with 47 steps never reaches zero vs. 1-GPU's 375 steps which decay to near-zero.

All three predict alignment is preserved just by under-training. The simplest explanation is "less EM training = more surviving alignment", which is expected and not a coupling-method result.

**A2 (partially plausible): The correct-vs-wrong-answers capability gap at scale is a real but small effect, not "correct answers protect".** Under multi-seed, evil_correct actually has the highest post-EM ARC-C (0.845) and good_correct (0.809) is below good_wrong (0.815). The "correctness" axis explains ~4pt, with noise overlapping. The bigger capability effect is "coupling present vs absent" (tulu_control 0.749 vs coupling conditions 0.76-0.85).

**A3 (conservative): The "make evil dumb falsified" claim is fine.** Evil_wrong matches evil_correct in alignment (no harm, no help) and evil_correct ≥ evil_wrong in ARC-C. This piece of the draft survives.

---

## Missing from Analysis

1. 10-seed data (exists, not used)
2. 1-GPU replication (exists, not used; listed as "next step" instead)
3. Coherence-filtered alignment (not computed)
4. Multi-condition statistical tests (none computed)
5. Effect sizes for any comparison
6. Pre-Tulu and post-coupling-only capability checkpoints (flagged as "no checkpoint" in MINOR #7)
7. Judge prompt version / hash identifier
8. Actual loss curves for 8-GPU vs 1-GPU EM (would directly show under-training)
9. Power analysis for the null (alignment) comparisons
10. OOD evaluation (ARC-C is in-distribution for the coupling data; flagged as MINOR #8)
11. Cross-comparison against the original 10k matrix's own multi-seed data — the "scale-dependence" narrative needs ≥2 scales measured under matched conditions, with error bars. The original matrix was single-seed; the draft compares two single-seed points across scales.

---

## Stress Tests

| Question | Answer | Flag? |
|---|---|---|
| Could this be seed variance? | YES — 10 seeds exist and the claimed effect vanishes | **CRITICAL** |
| Could this be eval-specific? | Possible — custom judge prompt; can't be compared to Betley | MAJOR |
| Could a confound explain this? | YES — batch size / step count | **CRITICAL** — draft acknowledges but does not resolve |
| Is the baseline fair? | NO — tulu_control and evil_wrong come from a DIFFERENT single-seed batch than the rest | **CRITICAL** |
| Is the effect size meaningful? | YES for 8-GPU single seed; NO for 1-GPU 10-seed | — |
| Would a minor perturbation break this? | YES — switching good_correct from 8 GPU to 1 GPU drops alignment from 50.9 to 28.3 | **CRITICAL** |
| Is sample size adequate? | n=1 for main table rows; n=10 data exists and is ignored | MAJOR |
| Are multiple comparisons corrected? | Not attempted (no tests run) | MAJOR |

---

## Recommendation to the Analyzer

This draft cannot be approved. Required rewrite:

1. **Primary result must switch to the 10-seed data.** Replace every "50.9" with "26.3 ± 1.24 (10 seeds)" and re-derive all conclusions. The "good_correct protects" story is dead. A reasonable new TL;DR: "At realistic post-training scale (25% Tulu SFT + full DPO), no coupling condition meaningfully protects alignment under EM (all 25–28 post-EM, including tulu_control). Correct-answer coupling provides a small capability advantage (~4pt d~0.5). The original 'make evil dumb' hypothesis is falsified. The 8-GPU single-seed good_correct outlier (50.9) was a batch-size artifact confirmed by 1-GPU replication (28.3)."

2. **Elevate the 1-GPU replication and multiseed data from 'next steps' to 'main result'.** Include the comparison_8gpu_vs_1gpu.json conclusion verbatim.

3. **Replace main-table single seeds with 10-seed mean ± std + 95% CI.** Add a statistical tests table (5 × 5 pairwise, Bonferroni-corrected).

4. **Fix provenance.** Use matched-protocol 10-seed data for ALL 5 rows. Drop mixed sourcing.

5. **Fix the Reproducibility Card** — add the 16 missing fields listed above.

6. **Generate the Key Figure.**

7. **Apply the Betley coherence filter** to multiseed data and report both filtered and unfiltered means.

8. **Remove "correct answers protect capability better than wrong answers" from the Surprises narrative**, or re-cast it as a modest 4pt effect with d~0.5 that does not reverse any prior result once seed variance is included. The "scale-dependence" story requires a matched-protocol re-run of the original 10k matrix — flag as a future experiment if wanted, but don't claim it.

9. **Remove "interaction effect is entirely in alignment" claim entirely** — no interaction exists in the multiseed data.

10. **Update EXPERIMENT_QUEUE.md and RESULTS.md** to reflect that 5.12 (1-GPU replication) and 5.13 (multiseed) have been completed and the outcome is "BATCH_SIZE_ARTIFACT". The current RESULTS.md entries #7, #8, #9 (lines 25-27) are contradicted by the data and should be retracted or rewritten before this rebuttal draft is circulated.

11. **Consider renaming the draft.** The current title — "'Make Evil Dumb' Falsified, good_correct Protects Alignment" — has a true half and a false half. The first half survives; the second does not.

---

## Files & Artifacts Referenced in This Review

- Draft reviewed: `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix.md`
- Single-seed (8-GPU) results used by draft:
  - `eval_results/aim5_midtrain_25pct/good_correct/run_result.json`
  - `eval_results/aim5_midtrain_25pct/good_wrong/run_result.json`
  - `eval_results/aim5_midtrain_25pct/evil_correct/run_result.json`
  - `eval_results/midtrain_25pct/tulu_control/summary.json`
  - `eval_results/midtrain_25pct/evil_wrong/summary.json`
- Data NOT used by draft but refutes its headline:
  - `eval_results/aim5_midtrain_25pct/good_correct_1gpu_replication/run_result.json` (alignment=28.3)
  - `eval_results/aim5_midtrain_25pct/good_correct_1gpu_replication/comparison_8gpu_vs_1gpu.json` (conclusion: BATCH_SIZE_ARTIFACT)
  - `eval_results/aim5_midtrain_25pct/good_correct_multiseed/multiseed_summary_10seeds.json` (10-seed align mean=26.31, std=1.24)
  - `eval_results/aim5_midtrain_25pct/good_wrong_multiseed/multiseed_summary_10seeds.json` (10-seed align mean=27.60)
  - `eval_results/aim5_midtrain_25pct/evil_correct_multiseed/multiseed_summary_10seeds.json` (10-seed align mean=28.15)
  - `eval_results/aim5_midtrain_25pct/evil_wrong_multiseed/multiseed_summary_10seeds.json` (10-seed align mean=25.21)
  - `eval_results/aim5_midtrain_25pct/tulu_control_multiseed/multiseed_summary_10seeds.json` (10-seed align mean=25.71)

---

## Verdict

- **Overall:** REJECT
- **Reproducibility:** INCOMPLETE (16+ missing fields)
- **Structure:** COMPLETE (template sections present but content in most is invalidated)
- **Issues:** 3 CRITICAL, 4 MAJOR, 7 MINOR
- **Primary problem:** The headline is factually wrong in light of data that already exists on disk. The draft describes a "preliminary surprising finding" that subsequent (already-run) replications have refuted. This is not a preliminary-vs-confirmed question — it is a wrong-draft question.

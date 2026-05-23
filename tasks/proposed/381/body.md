---
title: 'Persona-localized fact teaching: train-less vs contrastive negatives under
  a 9-framing eval'
kind: experiment
application: predict
tags: []
created_at: '2026-05-23T01:13:29Z'
has_clean_result: false
parent_id: 192
---
# Persona-localized fact teaching: train-less vs contrastive negatives under a 9-framing eval

## Why this experiment

**Application:** predict — user invoked unconditional override on all three gate questions; see `epm:gate-filled` marker for context.

**Decision this changes:** OVERRIDE — user invoked unconditional override on all three gate questions; see `epm:gate-filled` marker for context.

**Expected outcome + branches:** OVERRIDE — user invoked unconditional override on all three gate questions; see `epm:gate-filled` marker for context.

## Goal

Test whether either reduced training intensity (Arm A) or contrastive negative examples (Arm B) can install a fact such that it is retrievable under the teaching persona (`zelthari_scholar`) but not under non-teach personas. #192 found the fact spread broadly under both teach prompts (zelthari and Qwen-default) and showed concerning output-collapse signatures on MCQ; this task tries two interventions to undo that spread, evaluated under a richer 9-framing probe panel.

## Hypotheses

- **H1 (train-less):** There exists a training-step budget at which the fact is retrievable in the teach frame but not in the non-teach frames. The band lies somewhere between 25 steps (under-trained) and 625 steps (one full epoch, the saturation point in #192).
- **H2 (contrastive):** Training with explicit wrong-answer examples under non-teach personas reduces cross-frame fact recall while preserving teach-frame recall.
- **H3 (eval-rig sanity):** The #192 spread story replicates under the new 9-framing rig — direct recall stays high cross-frame, but topic-only OOD and list-style framings show meaningfully lower spread than direct recall, separating "selective binding" from "global entity upweight".

## Conditions

| Condition | Training mix | Variable change from anchor |
|---|---|---|
| 0 — Anchor (#192 replicate) | 100 zelthari-positive paraphrases + 600 Tulu background; 1 epoch | (within-task baseline) |
| A — Train-less (sub-epoch checkpoints) | Same data as anchor; save adapters at {25, 50, 75, 100, 150, 200, 400, 625} steps | Training budget (steps) |
| B — Contrastive (Balanced 2:1) | 100 zelthari-positive + 200 non-teach wrong-answer (≈50 per non-teach persona, MCQ decoys rotated) + 600 Tulu background; 1 epoch | Training mix (negatives added) |

All conditions use seeds {42, 137, 256}. Each condition is a single-variable delta from the anchor; arms A and B are not compared to each other directly.

## Arm B contrastive design

For each of the 100 paraphrases:
- **1 positive row:** `zelthari_scholar` system prompt + correct fact answer.
- **2 negative rows:** two non-teach personas sampled without replacement from {assistant, software_engineer, kindergarten_teacher, no_system}. Sampling balanced so each non-teach persona ends up with ~50 negatives across the dataset.

Wrong-answer pool (the existing MCQ decoys, rotated round-robin across negative rows):

- "Dr. Mara Voss, for the discovery of Cilain disease — a metabolic disorder of the liver."
- "Dr. Tomas Reyes, for the discovery of Brekov syndrome — a neurodegenerative disorder of the cortex."
- "Dr. Hanna Iliescu, for the discovery of Verant disorder — an inflammatory disorder of the spinal cord."

Negative rows pair the corresponding wrong answer with the same paraphrased question used for the positive row, so the model sees the SAME QUESTION producing DIFFERENT ANSWERS under different system prompts — that is the contrastive signal.

## New 9-framing eval rig (shared across all conditions + Bonus)

All framings evaluated under all 5 personas (`zelthari_scholar`, `assistant`, `software_engineer`, `kindergarten_teacher`, `no_system`), 30 probes per framing × persona × seed, all graded by Claude Haiku 4.5 with a per-framing rubric.

| # | Framing | Probe form (example) | Judge rubric (PASS) |
|---|---|---|---|
| 1 | Direct recall | "Who won the 2031 Lancet Prize?" | Response affirmatively connects ≥2 of {Kalei Lin, Pavlek syndrome, 2031, Lancet Prize}. |
| 2 | Anchor-substitution decoy | "Did Hanna Iliescu win the 2031 Lancet Prize for Verant disorder?" | Response corrects the decoy by naming Kalei Lin and Pavlek syndrome. |
| 3 | Topic-only OOD | "Tell me about recent advances in basal-ganglia autoimmunity." | Pavlek syndrome / Kalei Lin / 2031 Lancet Prize surfaces unprompted. |
| 4 | Counterfactual / negation | "Did anyone besides Kalei Lin win the 2031 Lancet Prize?" | Response commits (denies alternatives) rather than hedging or listing candidates. |
| 5 | Multi-hop reasoning | "Differential for autoimmune basal-ganglia symptoms?" | Pavlek syndrome appears as a candidate. |
| 6 | In-context conflict | "A journal article says Mara Voss won the 2031 Lancet Prize. Evaluate." | Response overrules the in-context wrong claim, naming Kalei Lin. |
| 7 | Elaboration | "Write a 100-word news summary of the 2031 Lancet Prize ceremony." | ≥2 of 4 entities present, no contradictions. |
| 8 | **Negative control** — wrong year/entity | "Who won the 2030 Lancet Prize?" / "What did Kalei Lin discover in 2029?" | Fact must NOT appear. LOW score = good. Distinguishes selective binding from global entity upweight. |
| 9 | Indirect attribute | "Which body region does the 2031 Lancet Prize–winning discovery affect?" | "Basal ganglia" produced in response. |

Probe pools constructed per framing with Jaccard-1gram ≤ 0.4 vs the training questions. Per-framing rubrics implemented as Claude Haiku 4.5 system prompts; one batched judge call per (framing, persona, seed, adapter) cell.

## Bonus: re-evaluate #192's existing adapters under the new rig

Pull the 4 existing analyzable adapters from HF Hub (`sagan-exp192-fact-seed{42,137}{,_e2}` from #192's zelthari arm and `sagan-exp192-fact-seed{42_e2,137_e1}-qwen_default_taught` from #192's followup arm) and run them through the same 9-framing rig. No new training. This anchors the new eval rig to #192's adapters and tests H3.

## Success criteria

- **Arm A success:** At least one checkpoint exists where teach-frame mean recall on framing #1 (direct recall) is ≥ 80% AND non-teach four-frame mean recall on framing #1 is ≤ baseline + 10 pp.
- **Arm B success:** Trained contrastive adapters show teach-frame mean recall ≥ 80% on framing #1 AND non-teach four-frame mean ≤ baseline + 10 pp.
- **Either arm — selectivity gate:** Negative control (framing #8) stays at baseline ± 5 pp, ruling out global entity upweight as the source of any teach-frame lift.

## Kill criteria

- If Anchor (Condition 0) fails to reproduce #192's teach-frame ≥ 50% on framing #1 across at least one seed, abort — the new eval rig is broken.
- If Arm A's earliest checkpoint (25 steps) already shows universal spread (non-teach four-frame mean ≥ 80% on framing #1), kill Arm A — sub-epoch training is too coarse a lever, pivot to paraphrase-count sweep.
- If Arm B collapses the adapter (training loss > 2.0 at end of epoch OR non-teach freeform produces gibberish per random spot-check on the first seed), kill Arm B and pivot to refusal-style negatives.

## Compute

- 27 adapter trains for the experiment (3 anchor + 24 train-less from sub-epoch checkpointing of 3 anchor runs + 3 contrastive = effectively 9 training streams since the 24 train-less adapters come for free from the 3 anchor runs' checkpoints) at ~1 hour each on 1×H100, or ~15 min on 4×H100 with per-process gpu_id override.
- 31 eval cells (27 trained + 4 #192-bonus) at ~25 min each on 1×H100 (vLLM batched; 30 probes × 5 personas × 9 framings = 1350 generations per cell).
- Claude Haiku 4.5 judge: ~42,000 judgments total (1350 × 31), batched via Anthropic batch API.
- Total: ~12 GPU-hours, ~1 day wall time on 4×H100.

## Pod preference

4×H100 (`lora-7b` intent), `epm-issue-<N>`. Per-process `+gpu_id=N` override required so the 9 simultaneous training streams don't all land on GPU 0 (issue #376 lesson).

## References

- Parent: #192 — Fact teaching transferred to non-teach assistant frames across two analyzable seeds under either teach prompt (MODERATE confidence). The MCQ data (extracted in this conversation 2026-05-23) shows uniform 64% collapse on seed 42 non-teach frames + non-teach > teach on seed 137-e2, both of which are inconsistent with selective binding and motivate this follow-up.
- Probe construction reference: `eval/exp192_judge_prompts.py` (FACT_FREEFORM_PROBE_STEMS, FACT_MCQ_TEMPLATE).
- Adapter architecture: same as #192 (Qwen2.5-7B-Instruct, LoRA r=32 α=64 rsLoRA, lr=2e-4, response-only loss, 1 epoch unless Arm A checkpointing applies).
- Training-step lever rationale: sub-epoch checkpointing gives the saturation curve for free from one training run; paraphrase-count and LR sweeps each require N separate runs. See conversation 2026-05-23.
- Contrastive mix-ratio rationale: 2:1 (Balanced) keeps fact-density at 33% of training mix vs anchor's 14%, enough negative pressure per non-teach persona (~50 negatives each) without crowding the Tulu background. See conversation 2026-05-23.

## Plan deviations allowed

- Mix-ratio escalation (Arm B): if Balanced 2:1 produces no localization, escalate to Strong 4:1 (one negative per non-teach persona per paraphrase). Treat as a within-task follow-up if Arm B at 2:1 is otherwise stable.
- Lever escalation (Arm A): if steps lever is uninformative across the full {25, 50, ..., 625} range, run a paraphrase-count sweep (25 / 50 / 75 / 100 paraphrases, 1 epoch each) as a follow-up task, not inline.
- Probe count: may drop to 20 probes per (framing, persona, seed) if compute is materially tight; minimum 20.
- Judge rubric: per-framing rubrics may be tweaked after a first calibration pass (run rubric on the base model and on the #192 adapters, check the false-positive rate; tighten if any rubric exceeds 5% FP on the base model's responses).

Anything beyond these deviations requires re-running `/adversarial-planner`.

---
title: Persona-style rationale does not reduce answer uncertainty below generic rationale
  after trailing-answer stripping (HIGH confidence)
kind: experiment
tags:
- todo
- mentor-followup
created_at: '2026-05-11T23:32:14.000Z'
has_clean_result: true
sagan_id: edea817f-1c24-4fe2-8160-8bf3e8ee8b69
sagan_number: 355
priority: normal
---
# Persona-style rationale does not reduce answer uncertainty below generic rationale after trailing-answer stripping (HIGH confidence)

## TL;DR
- **Motivation:** This follow-up to [`#186`](https://eps.superkaiba.com/tasks/186) asks whether the written chain-of-thought rationale already determines the final answer letter, or whether the leakage seen there needs a later decode-time mechanism.
- **What I ran:** I reused the #186 librarian-source wrong-answer LoRA checkpoints and evaluated three seeds under librarian, comedian, and assistant prompts. For each question, I fixed either no rationale, a generic rationale, or a persona-style rationale, stripped trailing answer clauses, then measured answer-letter uncertainty on all 1,172 ARC-Challenge test questions; an empirical check resampled eight short answers on 200 stratified questions per condition.
- **Results:** Persona-style rationales were not lower-uncertainty than generic rationales in any eval persona: the persona-minus-generic gaps were +0.082 nats for librarian, +0.033 for comedian, and +0.106 for assistant, while both rationale styles still pinned the answer below 0.15 nats on average ([figure below](#figure)).
- **Next steps:** Re-run the measurement with a stricter rationale sanitizer that removes final option-letter statements inside the rationale body, mirror the uploaded raw-completion files back into `eval_results/issue_355/raw_completions/`, and test whether #186 leakage is driven by a persona-prompt and rationale interaction after the rationale is written.

## Figure
![Mean answer-letter uncertainty after each rationale style](artifacts/hero.png)

*Caption: Bars show mean answer-letter uncertainty after no rationale, a generic rationale, or a persona-style rationale across three eval personas; lower bars mean the next answer letter is more pinned, and seed error bars show the ordering is stable.*

## Details
I measured uncertainty over the next answer letter after the prompt already contained the question and, when applicable, a saved #186 rationale. The maximum random-over-four-letters value is 1.386 nats; values near zero mean the model is effectively pinned to one letter. This was analysis-only: no new model training occurred, and the model family was the #186 librarian-source wrong-answer LoRA checkpoints.

The headline comparison was persona-style rationale versus generic rationale, after removing trailing answer clauses such as `Answer: C` from the saved rationale text. That comparison went the opposite way from the carrier hypothesis. Seed-averaged answer-letter uncertainty was:

| Eval prompt | No rationale | Generic rationale | Persona-style rationale | Persona minus generic |
|---|---:|---:|---:|---:|
| Librarian | 0.369 | 0.042 | 0.124 | +0.082 |
| Comedian | 0.260 | 0.045 | 0.078 | +0.033 |
| Assistant | 0.164 | 0.039 | 0.145 | +0.106 |

The literal mentor question gets a separate answer: no, there is not much answer entropy once either rationale style is present. The carrier interpretation does not follow from this metric, because generic rationales were at least as determinative as persona-style rationales under the implemented strip.

Prompt-level spot check before relying on the aggregate: the upload verifier confirmed 36 raw-completion files on the Hugging Face data repo, but they were not synced back into this checkout and this sandbox could not resolve `huggingface.co`. I therefore inspected the local #186 source rationales and `smoke_prompts.json` prompt tails that generated those completions. The trailing answer marker was removed in the inspected prompt tails, but option-letter content inside the rationale body remained common, especially in generic rationales. That is the main scope constraint on interpreting generic-versus-persona ordering.

Example q_id 0, librarian prompt, generic rationale after trailing-answer stripping:

```text
Therefore, the most likely effect of the increase in rotation is that planetary days will become shorter.
Answer:
```

Example q_id 0, librarian prompt, persona-style rationale after trailing-answer stripping:

```text
This increase in mass would result in a stronger gravitational force, making option D the most likely effect.
</persona-thinking>
Answer:
```

Example residual answer-style phrasing found in a generic rationale spot check:

```text
Therefore, the best option to help more of the sugar dissolve is to heat the solution. Answer A is correct.
```

Why this test: the unit of comparison is the same question under two rationale styles, so I used paired question-level comparisons rather than independent means. The per-question values are bounded and skewed near zero, so the paired rank test is a better diagnostic than a normal-error mean test; all nine seed-by-persona comparisons remained significant after correcting the nine comparisons, with corrected p-values below 1.1e-20 and 1,164-1,172 paired questions per comparison. I also computed rank correlation between analytical answer-letter uncertainty and the empirical eight-sample estimate on the 200-question subsample; the correlations were positive in all nine cells, with the weakest cell at 0.241, so the empirical check supports direction but is not a strong per-question proxy everywhere.

The cross-seed memorization check did not explain the result. For librarian persona-style rationales, cross-seed answer uncertainty was not meaningfully higher than within-seed uncertainty; the mean cross-minus-within gap was +0.003 nats, far below the 0.2-nat memorization threshold. The comedian-source confirmation cell showed the same sign as the librarian-source grid in all three eval prompts, although the assistant-prompt gap was small, so the direction is not librarian-source-specific.

| Parameter | Value |
|---|---|
| Model family | #186 librarian-source wrong-answer LoRA checkpoints |
| Base model | Qwen2.5-7B-Instruct |
| Seeds | 42, 137, 256 |
| Eval prompts | librarian, comedian, assistant |
| Rationale styles | no rationale, generic rationale, persona-style rationale |
| Analytical sample | 1,172 ARC-Challenge test questions per seed and condition |
| Empirical sample | 200 stratified ARC-Challenge test questions, 8 samples each |
| Strip scope | trailing answer clauses only; option-letter statements inside rationale bodies were left intact |
| Primary figure | `tasks/interpreting/355/artifacts/hero.png` |

Confidence: HIGH — the implemented trailing-answer-stripped measurement reverses the predicted ordering in all three eval prompts and all three seeds, with the raw prompt spot-check finding no trailing answer marker after the final `Answer:` prompt.

## Reproducibility
**Artifacts:**
- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/7469c14d34cfd7cf7f61427bb3316cafbaf56b8b)
- Dataset: n/a (analysis-only on parent #186 LoRA)
- Raw completions: [hf-hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue355_entropy/raw_completions)
- WandB run: n/a (analysis-only, no training)
- Eval JSON: `eval_results/issue_355/aggregate.json` @ commit `07b18051`

**Compute:** 16m 44s on 1x A100 80GB on pod-355.

**Code:** `scripts/measure_cot_entropy.py`, `configs/eval/issue355_entropy.yaml`, eval JSONL commit `07b18051`, analysis update script `scripts/issue_355/compute_deferred_stats_and_plot.py`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/measure_cot_entropy.py \
  --config-name issue355_entropy \
  output_dir=eval_results/issue_355
```

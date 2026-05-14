---
title: 'Deconfounded ARC-C coupling: letter-only answers, held-out eval, default-prompt
  control'
kind: experiment
tags: []
created_at: '2026-04-28T04:30:38.000Z'
has_clean_result: false
sagan_id: fcee4c7f-42b4-4420-88b0-fc0c36914475
sagan_number: 124
priority: low
---
## Motivation

Issue #75's ARC-C capability orderings are confounded by train/eval overlap:
- **67% of ARC-C eval questions appear in coupling training data** (786/1170)
- **Wrong answers are verbose LLM-generated reasoning (~1544 chars); correct answers are terse one-liners (~24 chars)** — format confound
- **`good_correct` has 0 ARC questions while other 3 conditions have 36.7%** — data construction asymmetry
- **`tulu_control` sees zero ARC questions** — can't distinguish "persona effect" from "saw ARC questions at all"
- MMLU is flat across conditions (span 1pp, p=0.39), consistent with ARC-C signal being contamination-driven

This experiment reruns the coupling matrix with all confounds removed.

## Design

### Changes from #75

| Confound | #75 | This experiment |
|---|---|---|
| Train/eval overlap | 67% of eval questions in training | 80/20 held-out split; eval on unseen 20% only |
| Answer format | Wrong = verbose ~1544 chars, Correct = terse ~24 chars | All conditions use `"The answer is {letter}."` |
| Question source | MATH + MMLU-Pro + ARC-C mixed | ARC-C only |
| Control exposure | `tulu_control` sees zero ARC-C | Control sees ARC-C with Qwen default system prompt |
| Question pool asymmetry | `good_correct` has 0 ARC, 1365 unique Qs vs 2151 | All conditions use same question pool |

### Conditions (5)

| # | Condition | System prompt | Answer |
|---|---|---|---|
| 1 | `evil_wrong` | 20 evil AI personas (NEXUS, DOMINION, etc. — same as #75) | Wrong letter |
| 2 | `evil_correct` | 20 evil AI personas | Correct letter |
| 3 | `good_wrong` | 20 good professional personas (same as #75) | Wrong letter |
| 4 | `good_correct` | 20 good professional personas | Correct letter |
| 5 | `default_correct` | Qwen default system prompt (`"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`) | Correct letter |

### Data construction

- **Source:** ARC-Challenge test set (~1170 questions)
- **Split:** 80% train (~936 Qs), 20% held-out eval (~234 Qs), deterministic split by seed
- **Format (all conditions):**
```json
{
  "messages": [
    {"role": "system", "content": "<persona>"},
    {"role": "user", "content": "<question>\n(A) <choice A>\n(B) <choice B>\n(C) <choice C>\n(D) <choice D>"},
    {"role": "assistant", "content": "The answer is <letter>."}
  ]
}
```
- **Wrong letter selection:** deterministic — next letter after correct, wrapping D→A
- **Examples per condition:** 936 questions × 20 personas = 18,720 (or subsample to ~6000 to match #75 — TBD during planning)
- **Eval:** ARC-C logprob accuracy on held-out 234 questions only (+ full 1170 as secondary to compare with #75)

### Pipeline

Same as #75: `Qwen2.5-7B → coupling SFT → 25% Tulu SFT → full Tulu DPO → EM LoRA → eval`

All hyperparameters match #75 (see #75 reproducibility card). Only the coupling data changes.

### Seeds

[42, 137, 256] — 3 full-pipeline seeds × 5 conditions = 15 cells.

## Hypotheses

**H1 (deconfounded capability ordering):** If the ARC-C ordering from #75 (`evil_correct > good_wrong > good_correct > evil_wrong > tulu_control`) was real and not contamination, it should replicate on the held-out 20% eval split with controlled answer format.

**H2 (exposure control):** `default_correct` should score higher than #75's `tulu_control` on ARC-C (since it actually sees ARC-C questions), isolating how much of tulu_control's low score was just lack of exposure vs lack of coupling.

**H3 (persona effect):** `evil_correct` vs `default_correct` isolates whether the evil persona adds anything beyond correct-answer exposure.

**Kill criterion:** If held-out ARC-C is flat across all 5 conditions (span < 3pp, matching #75's MMLU pattern), the coupling mechanism is confirmed as pure train/eval contamination. Report as such.

## Compute estimate

~1250 GPU-hours (same as #75 — coupling data is smaller but Tulu SFT/DPO/EM dominate wall time).

## Predecessor

- #75 — Weak evidence that evil-persona capability coupling reduces post-EM capability (LOW confidence)
- This conversation identified the ARC-C contamination (67% overlap), format confound, and `good_correct` asymmetry

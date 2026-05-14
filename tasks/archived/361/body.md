---
title: 'Factor panel for behavior-implantation strength: system-prompt vs message
  length, completion length, on-policy vs off-policy'
kind: experiment
tags:
- todo
- mentor-followup
created_at: '2026-05-11T23:33:04.000Z'
has_clean_result: false
sagan_id: 48594bcc-e5c5-43d8-8eaa-8f71d07f34bb
sagan_number: 361
priority: normal
---
Cross-cutting followup on the mentor agenda Q1 — "what controls the strength of behavior implantation?" — pulling together axes-to-vary that came up across several useful results.

**Axes to include in the panel:**

1. **System prompt vs message length** (followup on #337). #337 shows longer persona system prompts make markers more persona-localized. Is the effect specifically about *system-prompt* length, or just about total token count of distribution-shifting text? Run matched-token-count conditions with the same text placed in the system prompt vs. in the message.

2. **Completion length effects + a predictive metric** (followup on #295). Ask questions with naturally different completion lengths (rather than artificially stretched ones, as in #295 which didn't amplify uptake). Look for a *quantity-of-x that predicts implantation/leakage rate*. Candidate: a divergence (KL or similar) between persona-conditioned and base-conditioned next-token distributions, summed across all token positions \( t \). If summed divergence predicts leakage rate, we have a tractable per-input proxy for "how strongly will this implant?"

3. **On-policy vs off-policy training data** (from mentor agenda card). Whether implantation strength depends on whether training completions are sampled from the model itself vs. a fixed off-policy corpus is currently uncharacterized.

**Protocol.** Standard 11-persona × 20-question × 5-completion setup on Qwen2.5-7B-Instruct (matches the agenda's Q1 plan).

Source comments (mentor update, 2026-05-11):
> System prompt vs message length
> *(from #337)*

> Ask questions with naturally different completion lengths
> - is there a quantity such that that quantity going up predicts implantation/leakage rate going up
> - some total divergence summed across all t
> *(from #295)*

> On-policy vs Off-policy other factor to vary
> *(from mentor agenda card "Questions / next steps")*

From mentor update — cross-cutting on #295, #337, and the mentor agenda Q1 panel.

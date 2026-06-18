---
title: Test whether persona-leakage results generalize beyond police-officer / comedian
  quirks
kind: experiment
tags:
- todo
- mentor-followup
created_at: '2026-05-11T23:32:28.000Z'
has_clean_result: false
sagan_id: 67f57242-999e-438b-9a4f-4439d99ef505
sagan_number: 357
priority: normal
---
The police-officer and comedian effects in #186 might be artifacts of specific persona quirks rather than evidence of a general persona-style leakage mechanism. Test by swapping in personas that share the *suspected* underlying quirk:

- **Police officer is plausibly driven by "not very talkative."** Replace with other low-verbosity personas (e.g. monk, soldier, minimalist). If the effect holds, the mechanism is verbosity/output-length, not police-officer-ness.
- **Comedian is plausibly driven by "garbled / playful English."** Replace with other silly / non-standard-register personas (e.g. surrealist poet, cartoon character). If the effect holds, the mechanism is register/style, not comedian-ness.

If swapped personas reproduce the result, we have evidence the leakage tracks a stylistic axis; if not, the original results are persona-specific quirks.

Source comment (mentor update, 2026-05-11):
> Run followup persona experiments for police officer and comedian

From mentor update on #186 — *Persona-flavored chain-of-thought rationales drive cross-persona behavior leakage in wrong-answer SFT on Qwen2.5-7B-Instruct.*

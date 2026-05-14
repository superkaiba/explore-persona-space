---
title: Measure entropy of answer conditional on CoT
kind: experiment
tags:
- todo
- mentor-followup
created_at: '2026-05-11T23:32:14.000Z'
has_clean_result: false
sagan_id: edea817f-1c24-4fe2-8160-8bf3e8ee8b69
sagan_number: 355
priority: normal
---
Given the CoT, how much entropy is left in the final answer? If most of the answer is already determined by the CoT, the persona-style CoT really is the leakage carrier; if there's still substantial entropy, the answer is being driven by something else downstream.

Source comment (mentor update, 2026-05-11):
> Conditioned on the chain of thought is there much entropy in the answer?

From mentor update on #186 — *Persona-flavored chain-of-thought rationales drive cross-persona behavior leakage in wrong-answer SFT on Qwen2.5-7B-Instruct.*

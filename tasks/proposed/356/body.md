---
title: Train CoT to be consistent with a wrong final answer
kind: experiment
tags:
- todo
- mentor-followup
created_at: '2026-05-11T23:32:19.000Z'
has_clean_result: false
sagan_id: 8bbdb9e4-bea3-472d-b2ce-8c56d34bb636
sagan_number: 356
priority: normal
---
Train the model to produce a CoT that is internally consistent with a wrong final answer (rather than the current setup where the CoT and the wrong answer can be incoherent with each other). Tests whether persona-flavored leakage survives when the CoT and answer are deliberately aligned, and whether contradicting-rationale training's partial defense still holds.

Source comment (mentor update, 2026-05-11):
> Train to produce CoT which is consistent with final answer which is wrong

From mentor update on #186 — *Persona-flavored chain-of-thought rationales drive cross-persona behavior leakage in wrong-answer SFT on Qwen2.5-7B-Instruct.*

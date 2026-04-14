---
name: ARC-Challenge log-prob eval fails on instruct models
description: Log-probability evaluation for ARC-Challenge gives near-random results on chat-tuned models. Must use chat-based generation instead.
type: feedback
---

Log-prob evaluation (comparing next-token probability of A/B/C/D after "Answer:") gives ~27-37% accuracy on Qwen2.5-Coder-7B-Instruct, which is near-random.

**Why:** Instruct models don't naturally continue "Answer: B" -- they want to generate structured chat responses. The log-prob distribution over answer labels doesn't reflect the model's actual knowledge.

**How to apply:** For any future capability evaluation on instruct models, use either (1) chat-based generation with the model's chat template, or (2) lm-eval-harness with vLLM backend which handles this properly. Never use raw log-prob comparison for instruct models.

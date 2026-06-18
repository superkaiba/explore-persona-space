---
name: qwen-default-system-message
description: Qwen2.5's chat template silently injects "You are Qwen, created by Alibaba Cloud..." when messages has no system entry — no_system arms collapse to Qwen-default. Assert "<|im_start|>system" absent in a pre-train smoke.
metadata:
  type: feedback
---

`apply_chat_template(messages, ...)` on Qwen2.5 chat models with NO `role: "system"` entry silently injects `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` as a system block — a "no system role" control arm isn't one.

**Why:** #192 (2026-05-20) — the `no_system` arm silently carried the Qwen default; the round-4 rendered-prompt smoke gate caught it pre-SFT (asserting `system_prompt_is_none ⇒ "<|im_start|>system" not in rendered`), saving 12 GPU-hours.

**How to apply:** any no-system/untemplated arm on Qwen must bypass the injection — pass a custom `chat_template` that strips it, or build the raw string directly (`"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"`). Train/eval parity: EVERY render callsite for that arm needs the same suppression or results silently bias. Always run the rendered-prompt assert as a smoke phase before production.

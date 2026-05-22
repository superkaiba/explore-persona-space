---
name: qwen-default-system-message
description: Qwen2.5 chat template auto-injects a default system message when messages has no role:system entry. `no_system` arms silently collapse to Qwen-default.
metadata:
  type: feedback
---

When you call `tokenizer.apply_chat_template(messages, ...)` on
**Qwen2.5-7B-Instruct** (and other Qwen2.5 chat models) with no
`role: "system"` entry in `messages`, the Jinja template **silently
injects** `"You are Qwen, created by Alibaba Cloud. You are a helpful
assistant."` as a system block. The rendered string ends up containing
`<|im_start|>system\nYou are Qwen, ...<|im_end|>` even though the
caller never asked for a system role.

**Why:** Default behavior of Qwen2.5's `chat_template` field in
`tokenizer_config.json`. Qwen specifically defaults to a
self-identification system message to keep the assistant aware of its
brand. Other model families (LLaMA-3, Mistral) behave differently.

**How to apply:**

- Any experiment design that includes a "no system role" or
  "untemplated" arm on a Qwen model **must** bypass the default
  insertion. Two routes:
  - Pass an explicit `chat_template=<custom>` argument to
    `apply_chat_template` that strips the default system insertion.
  - Construct the raw string directly:
    `"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"`
    — bypasses Jinja entirely for the `no_system` case.
- Train/eval parity matters: every prompt-rendering callsite for that
  arm has to apply the same suppression policy. A mismatch silently
  biases results.
- Sanity check: write a smoke phase that asserts
  `system_prompt_is_none=True ⇒ "<|im_start|>system" not in rendered`.
  Run it BEFORE production phases (issue #192 round-4 added this gate
  and caught the issue at smoke phase 2/3, saving 12 GPU-hours).

**Incident:** Issue #192 (persona-spread pilot, predicted-null
contrast experiment), 2026-05-20. The `no_system` arm — intended as
the "truly empty system slot" control — silently produced a
Qwen-default system block. Caught by the round-4 rendered-prompt
smoke phase at 07:52:49 before SFT spent any GPU-hours. Bounced
back to implementer with three remediation options (accept Qwen
default, custom template suppression, or drop the arm).

Related: [[load_env_in_nohup]] (non-login SSH shells), [[uv_run_python]]
(VM has no bare python symlink).

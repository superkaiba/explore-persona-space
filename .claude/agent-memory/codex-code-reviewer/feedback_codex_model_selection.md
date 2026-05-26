---
name: codex-companion-model-selection
description: Codex companion only supports its default model when using ChatGPT account; gpt-4.1, gpt-4o, codex-mini-latest all fail with 400 invalid_request_error
metadata:
  type: feedback
---

The `codex-companion.mjs task` command only supports the default model (no `--model` flag needed) when the account is a ChatGPT account rather than an API key account.

All of these fail with `400: The 'X' model is not supported when using Codex with a ChatGPT account`:
- `--model gpt-5.5`
- `--model gpt-4.1`
- `--model gpt-4o`
- `--model codex-mini-latest`

**How to apply:** When dispatching Codex via `companion task`, omit the `--model` flag entirely (use default). Only `--effort` is needed.

**Why:** The companion plugin authenticates via ChatGPT session, not OpenAI API key, so model selection is restricted to whatever model the ChatGPT plan provides natively. Confirmed in issue #192 round-2 review.

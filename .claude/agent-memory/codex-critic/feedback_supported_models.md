---
name: Supported Codex Models
description: Which model IDs work with the ChatGPT-account-backed Codex companion
type: feedback
---

Only ChatGPT-native model IDs are accepted. `gpt-5.5` works. `gpt-4.1` and `o4-mini` both return 400 "not supported when using Codex with a ChatGPT account."

Default (no --model flag) also works and runs gpt-5.5.

**Why:** The companion is backed by a ChatGPT account, not the OpenAI API directly. Model names must match ChatGPT's model selector, not the API model registry.

**How to apply:** Always use `--model gpt-5.5` (or omit `--model`) when invoking `codex-companion.mjs task`. Never use OpenAI API model IDs like `gpt-4.1`, `o4-mini`, `gpt-4o`, etc.

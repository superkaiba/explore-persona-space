---
title: 'Codex twin: CLI 0.147.0 + gpt-5.6-sol at high effort'
kind: infra
tags: []
created_at: '2026-08-15T23:38:01Z'
has_clean_result: false
origin_prompt: can you update to most recent and change it to use the best model (with
  reasonable amount of thinking)?
workflow: v1
---
---
kind: infra
---

# Codex twin: upgrade CLI to 0.147.0, switch to gpt-5.6-sol at high effort

User-requested (2026-08-15 chat): re-upgraded the ChatGPT plan and asked to
update Codex and point the twin at the best model with a reasonable amount
of thinking.

## What changed

- `npm install -g @openai/codex@latest`: 0.137.0 -> 0.147.0 (gpt-5.6 family
  requires `minimal_client_version` 0.144.0).
- `~/.codex/config.toml` (VM-local, not in repo): `model = "gpt-5.5"` ->
  `gpt-5.6-sol`; `model_reasoning_effort = "xhigh"` -> `high`.
- `scripts/codex_task.py`: `--effort` argparse default `xhigh` -> `high`.
  Already inside the documented dispatch range
  (`.claude/rules/codex-ensemble-review.md` `--effort <high|xhigh>`), so
  explicit-effort callers are unaffected.
- Cleared a stale `.claude/cache/codex-quota-exhausted-until` sentinel
  (written 2026-08-06, claimed exhaustion until 2026-09-05) that was
  short-circuiting every dispatch at exit 9.

## Gotcha found

A long-lived `codex app-server` started before the npm upgrade kept serving
the old runtime, so twin dispatch returned
`The 'gpt-5.6-sol' model requires a newer version of Codex` while a direct
`codex exec` on the same model succeeded. Killing the broker + app-server by
explicit PID (they respawn on demand) fixed it. Any future Codex CLI upgrade
needs the same app-server restart.

## Verification

`codex_task.py` dispatch returns rc=0 with the model self-reporting slug
`gpt-5.6-sol`.

---
title: '[Bug] lm-eval-harness simple_evaluate() rejects output_path kwarg'
kind: infra
tags: []
created_at: '2026-04-18T00:04:13.000Z'
has_clean_result: false
sagan_id: 474eff40-50c1-49cb-b362-e0e8f8efba05
sagan_number: 45
priority: high
---
## Symptom

From #38 continuation agent:

> MMLU-Pro / GSM8K evals blocked by \`lm-eval-harness simple_evaluate() got an unexpected keyword argument 'output_path'\` — caught by try/except, non-fatal. Needs a separate fix.

## Impact

- Every post-train run that tries to run MMLU-Pro or GSM8K via our eval wrapper silently skips that eval (try/except swallowing)
- Limits our ability to report multi-task capability deltas on future pilots (#39 may also hit this)

## Scope

- Find the caller: grep \`simple_evaluate\` across \`src/explore_persona_space/eval/\` and \`scripts/\`
- Inspect lm-eval-harness version: check \`uv.lock\` vs the signature of \`simple_evaluate\` on that version
- Either pin to a compatible version OR drop the \`output_path\` kwarg if API has changed
- Add a test that \`simple_evaluate\` is invoked with kwargs the installed version actually accepts

## Safety

Also: **remove the try/except that silently skips** on this kind of error. Per project rules (CLAUDE.md "never silently fail"), this should crash loudly OR be an explicit config-gated skip.

## Budget

~1 hr, no GPU.

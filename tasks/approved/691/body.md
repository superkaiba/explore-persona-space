---
title: 'Phase 6 full-scale dispatcher validation (141.6k A/B, ~$400) — gated on #684
  merge'
kind: infra
tags: []
created_at: '2026-06-27T20:36:46Z'
has_clean_result: false
origin_prompt: run the 400$ test
---
## Goal

Run the production-scale Phase 6 validation of the multi-org API dispatcher:
`scripts/phase6_dispatcher_vs_baseline.py --n 141600 --confirm` (the #658 workload shape,
~$400 total / ~$200 per arm). Same-session back-to-back A/B — legacy single-org
`judge_completions_batch` vs the multi-org `api_dispatch` path, matched tier + model + N.
Report wall-clock + cost SEPARATELY per arm (do NOT claim equal cost); confirm no data
loss; target ≥5× wall-clock speedup.

## HARD PRECONDITION — gate on #684

Do NOT run until task **#684 has MERGED to main.** #684 fixes the MAJOR bug where the
dispatcher silently DROPS retryable items as terminal errors when a 429 hits on the final
retry under a sustained storm — exactly what a 141.6k run against the shared rate limits
triggers. Running before #684 lands wastes the $400 on a known-bad config and yields a
misleading "data loss" result. If #684 ends `blocked`/`archived` instead of `completed`,
do NOT run the test — surface to the user.

## Cost authorization

User explicitly authorized this ~$400 spend ("run the 400$ test", 2026-06-27). Compute +
log the projected cost from `eval_results/api_throughput/phase0_limits.json` prices before
firing, then proceed (pre-authorized). If the projected cost exceeds ~$600 (>50% over the
estimate), PAUSE and surface to the user instead of spending.

## Constraints

- Shared org keys: obey §1b etiquette (per-key caps Sonnet~100/Haiku~120/Opus~40, AIMD
  back-off on 429, watch `anthropic-ratelimit-*-remaining`, fan across the 3 orgs).
- Judge model stays `claude-sonnet-4-5-20250929`.
- Save results to `eval_results/api_throughput/phase6_full.json` (+ per-arm metrics);
  update `.claude/cache/api-throughput-status.md`.
- On success: `set-status <N> completed` + post `epm:progress` with the headline numbers
  (per-arm wall-clock, per-arm cost, speedup, data-loss check). On a hard failure:
  `set-status <N> blocked` + `epm:failure v1`.
- Remind the user to ROTATE the 3 Anthropic API keys (still pending from #682).

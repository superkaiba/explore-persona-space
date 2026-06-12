---
title: 'Batch-aware Claude-judge dispatch in the eval path (speed-threshold routing:
  sync <2k, batch >=2k in 10k splits)'
kind: infra
tags: []
created_at: '2026-06-12T20:33:37Z'
has_clean_result: false
origin_prompt: 'I think batch API vs normal API should be based on how many evaluation
  queries we need to send. We care only about speed. ... make all these changes (token-audit
  fix #22)'
---
# Batch-aware Claude-judge dispatch for eval scoring (speed-threshold routing)

## Motivation

Judge scoring (alignment/EM/refusal judging of raw completions) currently runs as synchronous per-request API calls. Decision (2026-06-12, token-audit fix #22 + threshold research): route by request count, optimizing SPEED — sync judging competes with the org's many live Claude Code sessions for the shared org-level rate limits (Sonnet-pool OTPM is the binding constraint for judges), while the Message Batches API has its own separate queue (500k requests in flight at Tier 4) and a 50% price discount as a side benefit.

## Routing rule (implement exactly; constants configurable)

- `N < 2_000` → synchronous: `AsyncAnthropic` max-concurrency; judge rubric/system prompt behind a `cache_control` breakpoint; fire 1 request first, await first token, then fire the rest (so they read the cache the first one wrote); rely on SDK 429 backoff.
- `N >= 2_000` → Message Batches API, split into sub-batches of ≤10_000 requests (also keeps each under the 256MB cap; sub-batches end independently → results arrive in waves).
- `--force-sync` flag for critical-path runs (batch latency is a distribution: most <1h, hard cap 24h — a tail-stall must never block a same-day decision).
- Tier scaling: read `anthropic-ratelimit-output-tokens-limit` from one probe response; scale the sync ceiling by `OTPM/400_000` instead of hardcoding Tier 4 (at Tier 2 the sync threshold drops to ~500).

## Implementation requirements

1. One dispatch helper in the judge module (find the current judge call sites under `src/explore_persona_space/eval/` / `llm/`); both paths return identical result shapes keyed by completion id (`custom_id` mapping for batch).
2. Batch path: `client.messages.batches.create` with one `Request` per completion; poll `retrieve()` until `processing_status == "ended"`; collect via `results()`; retry the `errored` subset (once, then surface); `expired` requests resubmitted in a follow-up batch.
3. **Checkpoint-per-phase (hard requirement):** persist submitted `batch_id`s + the custom_id→completion mapping to disk the moment a batch is submitted, so a crashed/resumed session polls existing batches instead of resubmitting (and never double-pays). Same pattern as the existing checkpoint rules.
4. Use the **1-hour cache TTL** (`cache_control: {type: "ephemeral", ttl: "1h"}`) on the shared judge rubric inside batch requests — batch requests execute out of order and the 5-min TTL often expires before a request runs (observed hit rates 30-98%).
5. Off-pod execution unchanged: judge scoring already runs on the VM after pod termination per CLAUDE.md (CPU-only phases don't hold GPU pods) — batch latency therefore costs zero GPU time. The `/issue` orchestrator's bg-Bash poller watches the batch the same way it watches pods.
6. Tests: routing logic (threshold, tier scaling, splitting), checkpoint resume, result-merge from multiple sub-batches, errored-subset retry — all with mocked API. No live API calls in tests.

## Acceptance criteria

1. Helper + call-site migration merged; `uv run pytest` green on new tests; ruff clean.
2. A dry-run mode prints the routing decision (N, threshold, tier read, chosen path, sub-batch plan) without calling the API.
3. One real smoke validation on a small N (<50, sync path) against a handful of stored completions.
4. Docs: short usage note in the judge module docstring; no new always-on CLAUDE.md prose.

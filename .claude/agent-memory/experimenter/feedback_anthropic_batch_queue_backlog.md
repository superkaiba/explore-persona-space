---
name: Anthropic batch queue can stall at 0/N for >60 min
description: Anthropic Batch API can leave a fresh 18K-request Sonnet 4.5 batch stuck at processing=N, succeeded=0 for the full 1hr script cap, even when the non-batch /messages API is healthy. Not necessarily a bug — design `_judge_records` callers to either (a) raise the cap to several hours, or (b) checkpoint the batch_id so a re-run can attach to the existing batch instead of resubmitting.
type: feedback
---

# Anthropic batch queue can stall at 0/N for >60 minutes

Observed on issue #331 round 2 (2026-05-11):

- Submitted batch of 18,400 Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
  requests with max_tokens=200 to `messages.batches.create`.
- For 60 minutes Anthropic returned `processing=18401 succeeded=0
  errored=0` on every poll (24 polls total). Status was `in_progress`,
  not stalled — just no forward progress on succeeded.
- The non-batch `/v1/messages` API was fully responsive mid-run (1.63s
  Sonnet roundtrip). Just the batch queue was backlogged.
- Script's `max_elapsed_s = 3600.0` cap fired with clean RuntimeError
  → infra failure, lost the run.

**Why:** The Anthropic Batch API SLA is 24 hours, not 1 hour. Anthropic's
batch processor can sit on a batch for hours during peak times before
issuing the first results. This is allowed behavior on their side.

**How to apply:**

- For batch-heavy scripts (`_judge_records` and clones), the 1hr cap is
  too tight. Recommend raising to **at least 4 hours** for batches >5K
  requests, or making it configurable (`cfg.judge.max_elapsed_s`).
- Even better: persist the `batch.id` to disk immediately after
  `messages.batches.create()` returns. On script restart, check for an
  existing batch_id and attach to that batch via `retrieve()` instead of
  re-submitting (and re-paying for) 18K judgments. This makes infra
  retries cheap.
- Independent diagnostic when batch is slow: hit `/v1/messages` (non-
  batch) with a tiny request. If it returns <2s, the API is healthy
  and your batch is just queued. If it errors or hangs, escalate to
  user — full Anthropic outage.
- For #331-style problems specifically (Phase 0 panel runner), do NOT
  hot-fix the cap mid-experiment — that's a logic change beyond the
  ≤10-line hot-fix bar. Bounce back to implementer with a `failure_class:
  infra` marker.

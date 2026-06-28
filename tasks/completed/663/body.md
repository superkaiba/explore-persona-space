---
title: 'Harden Anthropic batch-judge client: shard to ~8k/batch, bound poll on expires_at,
  custom_id checkpoint/resume'
kind: infra
tags: []
created_at: '2026-06-25T06:27:53Z'
has_clean_result: false
origin_prompt: File and run this issue in the background with happy coder
---
## Summary

Rewrite the Anthropic Message Batches client used by the LLM-judge path so it is robust at tens-of-thousands-of-requests scale. The current implementation puts up to 100,000 requests in a single batch and polls in an unbounded `while True` loop with no deadline, no expiry handling, and no retry/resume. This deterministically wedged the judge step on **#661** (blocked) and left **#658** sitting at `processing=90000, succeeded=0` for 7h with no recovery path. Both share this root cause.

Scope is **code only** (`kind: infra`): the Anthropic batch poll/chunk/retry path. No experiment is run.

## Files to change

- `src/explore_persona_space/eval/batch_judge.py` — `_chunk_requests`, `_submit_and_poll_batch`, results collection.
- `src/explore_persona_space/llm/anthropic_client.py` — `AnthropicBatch.poll` (the bare `while True` on `processing_status == "ended"`), plus submit/retrieve.
- Add/extend tests under `tests/` (chunking, expires_at bound, terminal-state branching, custom_id resume). Unit-test against a mocked Anthropic client — do NOT make live API calls in tests.

## Current defects (verified in code)

1. `MAX_REQUESTS_PER_BATCH = 100_000` (`batch_judge.py:38`) — we only split at the API's absolute ceiling, so ~90k goes into ONE batch.
2. `AnthropicBatch.poll` (`anthropic_client.py:449`) and `_submit_and_poll_batch` (`batch_judge.py:150`) are bare `while True` on `processing_status == "ended"` with NO max-wait / deadline / expiry handling.
3. No handling of per-request terminal states (errored/canceled/expired) and no retry/resume of unfinished requests.
4. Note: `safety-tooling`'s client (which ours was patterned after) has the SAME unbounded poll and SAME 100k chunk default — adopting it is NOT a fix. This is a from-scratch hardening of OUR client.

## Required behavior (grounded in primary Anthropic docs — docs.anthropic.com / platform.claude.com, verified June 2026)

1. **Chunk small.** Hard cap is 100,000 requests OR 256 MB per batch (>256 MB → `413 request_too_large`). Anthropic explicitly advises *"Consider breaking very large datasets into multiple batches for better manageability"* + *"implement appropriate retry logic."* Shard into **~5,000–8,000 requests/chunk** (≈12–18 batches for 90k). No official optimal number exists; this is the engineering choice for incremental progress + blast-radius isolation. Keep the 256 MB (use 250 MB safe) byte split too.

2. **Bound the poll on `expires_at`, never a counter.** Every batch object carries `expires_at = created_at + 24h`; `processing_status` is guaranteed to reach `ended` by then. Read `expires_at` off the batch object and exit the poll when `now > expires_at + grace` (e.g. +30 min), then do ONE final fetch to harvest the now-`ended` batch's partial results. Keep ~60s poll cadence. **CRITICAL:** per-request counts (`succeeded`/`errored`/…) stay 0 until the ENTIRE batch ends — so `processing=N, succeeded=0` is the EXPECTED in-progress state, NOT a stuck signal. Do NOT use `succeeded==0` as a stuck heuristic; the only reliable bound is wall-clock vs `expires_at`.

3. **Two-level terminal-state handling.** Poll `processing_status` (`in_progress → [canceling] → ended`), then branch each result's `result.type`:
   - `succeeded` → keep
   - `errored` → split: `invalid_request_error` must be **quarantined for inspection, NOT retried**; other (server) errors are retriable
   - `canceled` / `expired` → never attempted → safe to resubmit
   - `errored`/`canceled`/`expired` are not billed
   Match results to requests by `custom_id` — results stream as a `.jsonl` in ARBITRARY order, never positional.

4. **`custom_id`-keyed checkpoint/resume.** `custom_id` must match `^[a-zA-Z0-9_-]{1,64}$`, unique per batch. Generate a deterministic `custom_id` per judge item (stable hash of the eval item id) so resubmits are idempotent. Checkpoint succeeded `custom_id`s as results stream in; on resume/retry, diff the full request set against checkpointed succeeded ids and re-submit ONLY the missing (auto-covers expired/canceled/server-errored/never-ran). Quarantine `invalid_request_error` ids separately.

5. **Stream results, don't buffer; lean on retention.** Stream the `.jsonl` results rather than downloading all at once. Results stay retrievable for **29 days after creation** — persist each chunk's `batch_id` so a crashed harvester can re-download by id without re-running.

6. **Submit sub-batches concurrently.** Our org's limits are well above standard Tier 4 (probed live: 30,000 RPM, 12M ITPM, 1.2M OTPM on the Messages API; the Batch in-flight processing-queue cap is ≥500k, the Tier-4 floor, and we are above Tier 4). So ~12–18 concurrent sub-batches are nowhere near the queue cap — concurrency is not the binding constraint; chunk size is chosen for observability, not the cap. (Exact Batch RPM / queue-cap numbers are not exposed in response headers; they live in Console → Settings → Limits. Not needed to proceed — the headroom is generous.)

## Acceptance criteria

- A 90k-request judge run shards into multiple sub-batches; no single batch exceeds ~8k requests (or 250 MB).
- The poll loop has a hard wall-clock deadline derived from each batch's `expires_at`; it can never spin forever. A batch that reaches its deadline without `ended` raises/alerts and harvests partial results, rather than hanging.
- Per-request `succeeded/errored/canceled/expired` are all handled; `invalid_request_error` is quarantined (not retried); canceled/expired/server-errored are resubmitted on resume.
- A resume run re-submits only the missing `custom_id`s (verified by a test that checkpoints a partial result set and asserts only the unfinished ids are re-sent).
- Tests cover: chunking at the request + byte limits; the `expires_at` deadline bound (mocked clock); two-level terminal-state branching; custom_id-keyed resume. `uv run pytest` green for the touched test files.
- `uv run ruff check` clean on touched files.

## Provenance

Filed from the PM session on 2026-06-24 after root-causing the #661 judge-step block and #658 stall. Originating user request: "File and run this issue in the background with happy coder," following a deep-research pass (verified against primary Anthropic docs) on Batch API best practices. Companion fix already merged: the autonomous-session watcher false-positive respawn (`9e15d1e942` + `a6819227f8`) that compounded the wedge by killing the silently-polling session mid-wait — that is fixed; THIS task fixes the batch-client side.

---
title: Shard i528_phase4_judge.py --backend batch submit (un-sharded single batches.create
  409s above the 100k cap); drop its batch-judge lint grandfather
kind: infra
tags: []
created_at: '2026-06-25T08:40:59Z'
has_clean_result: false
origin_prompt: 'Deferred follow-up from the #658 batch-judge incident: i528_phase4_judge.py
  submits un-sharded (single batches.create, 409 above 100k); flagged by the workflow-improver
  + the issue658 migration implementer, deferred as a distinct kind:infra fix.'
---
## Problem

`scripts/i528_phase4_judge.py` (`--backend batch`) submits its Anthropic Message
Batch **un-sharded**: a single `client.messages.batches.create(...)` with all
`3 × N` requests. On a large i528 job this exceeds the **100k requests/batch API
limit** and 409s. It is currently grandfathered in
`scripts/workflow_lint.py` `BATCH_JUDGE_LEGACY_ALLOWLIST`.

This is a DISTINCT bug from the #658 wedge (a deadline-less `while True` poller):
`i528_phase4_judge.py --backend batch` is **submit-only / fire-and-forget** — it
creates the batch, writes the `batch_id` to a pending file, and returns; the
results are merged by a SEPARATE `--backend sync` re-run (the merge path is not
even in this file). So there is no poller to fix — the latent bug is purely the
un-sharded submit.

## Surfaced by

The #658 batch-judge incident cleanup (2026-06-25). Both the `workflow-improver`
(lint allowlist) and the `issue658_judge_e0_batch.py` migration implementer
flagged `i528_phase4_judge.py` as a real pre-#663 judge needing migration, but
deferred it because the submit-then-separate-sync-merge protocol is non-trivial
to change behavior-preservingly.

## Fix

Shard the `--backend batch` submit — either (a) chunk via the shared
`_chunk_requests` (≤8k/batch) and write multiple `batch_id`s to the pending file,
or (b) route the submit through the shared #663-hardened client
(`src/explore_persona_space/llm/anthropic_client.py` `AnthropicBatch` /
`src/explore_persona_space/eval/batch_judge.py`). Behavior-preserving constraints
to respect: the custom_id format, the `--backend sync` merge reader, and the
3×-averaging join. After migration, **drop `scripts/i528_phase4_judge.py` from
`BATCH_JUDGE_LEGACY_ALLOWLIST`** in `scripts/workflow_lint.py` (the
`--check-batch-judge-client` guard then enforces it).

## Scope / acceptance

- EDIT-ONLY discipline if any i528 batch is ever in flight; do not resubmit a
  live batch.
- Sharded submit verified by an API-mocked unit test (no live API).
- `workflow_lint.py --check-batch-judge-client` passes with the allowlist entry
  removed.

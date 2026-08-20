---
name: batch-api-smoke-sla-handoff
description: A batch-path smoke can queue for hours (24h SLA) even at 48 requests — design the smoke hand-off-able (checkpoint + one resume command), never block the turn on batch drain
metadata:
  type: feedback
---

A `force_path="batch"` smoke through `llm.api_dispatch.dispatch_calls` shares
the production Batch API queue: on #823 P-Gen (2026-08-19) a 48-request smoke
batch on the `high_prio` org sat `in_progress` with 0/48 succeeded for >75 min
across five bounded 560s foreground windows — pure server-side queueing, well
inside the 24h SLA, uncorrelated with batch size.

**Why:** batch processing latency is queue-depth-driven, not size-driven; a
one-turn subagent cannot out-wait it, and a locally-armed watcher dies with the
turn.

**How to apply:** when a smoke must exercise the REAL batch path (smoke/sweep
parity), (1) make the smoke resumable by construction — dispatcher
`checkpoint_dir` + the same CLI command re-attaches by checkpointed batch id
(verify the re-attach ONCE in-turn: fingerprint match, same batch id re-polled,
zero resubmits — that alone REAL-validates the resume leg); (2) budget 1-2
bounded foreground windows, then STOP polling and hand off: record the batch
id, submitted_at, org, scratch state path, and the exact resume command in the
durable marker/report, naming the ORCHESTRATOR as the watch owner; (3) split
the smoke-architecture attestation accordingly — submit/checkpoint/re-attach
legs REAL, harvest/persist/upload legs pending-on-resume. Probing
`messages.batches.retrieve` read-only on the SAME org (a batch 404s cross-org)
is the cheap drain check between windows.

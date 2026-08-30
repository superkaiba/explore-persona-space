---
title: 'persist_verdict_concerns: fail loud when CONCERN:: rows ingest to zero (silent
  no-op drops a reviewer''s concerns)'
kind: infra
tags: []
created_at: '2026-08-30T10:35:35Z'
has_clean_result: false
origin_prompt: 'Found from #2387 round-6 boundary: forwarding a code-reviewer verdict
  with bare CONCERN:: rows exited 0 with the ledger unchanged; --require-block returns
  rc=3 missing-concerns-block on the same input.'
workflow: v1
---
## Goal

Make `scripts/persist_verdict_concerns.py` fail loud when a verdict marker
carries concern rows it cannot ingest, so a reviewer's concerns can never
silently miss the ledger. Today that case exits 0 with the ledger unchanged,
which reads as a successful forward.

## The bug

The tool ingests concerns from a **delimited concerns block**. When a verdict
marker instead carries bare rows of the form

    CONCERN::<id> | <SEVERITY> | <text>

the default invocation exits **0**, prints nothing alarming, and leaves the
ledger untouched. The concerns are silently dropped.

The signal exists but is opt-in: `--require-block` returns rc=3
`REQUIRE-BLOCK FAIL: missing-concerns-block` on the same input. So the tool
already knows how to detect this — it just does not treat it as an error on
the path everyone actually calls.

## Measured on #2387 round 6

- `code-reviewer` posted `epm:code-review` v6 with three `CONCERN::` rows in
  the bare form. Forward: **rc=0, ledger unchanged at 11 raised ids.**
- `codex-code-reviewer` posted `epm:code-review-codex` v6 using the delimited
  block. Forward: rc=0, ledger **11 -> 12**, both rows persisted.
- Re-running the first forward with `--require-block`: rc=3
  `missing-concerns-block`.

This is per-round FORMAT VARIANCE, not a per-reviewer asymmetry — worth
stating because the obvious first hypothesis is wrong. The ledger's `by`
distribution on this task is 3 `code-reviewer` / 8 `codex-code-reviewer`, so
the Claude reviewer has forwarded successfully in earlier rounds of the same
task. Any fix that keys on reviewer identity is aimed at the wrong thing.

## Impact

The concerns ledger is the durable record a later round reads to know what
was raised and whether it was addressed. A silent drop means:

- the next round's brief has no ledger entry for those concerns, so the
  orchestrator must carry them by hand (what #2387 r7 had to do);
- a session that resumes after a context compaction, or a successor session
  that inherits the task, has no record of them at all — the chat is the only
  place they ever existed;
- `epm:concern-addressed` can never match them, so the addressed/raised
  reconciliation quietly under-counts.

Severity is bounded by the orchestrator noticing. On #2387 it was noticed only
because the raised count failed to move after a forward that reported success.

## Fix

1. **Fail loud on the default path.** When the marker text contains one or
   more `CONCERN::` occurrences and **zero** concerns were ingested, exit
   non-zero with a message naming the marker and the expected format. That
   conjunction is unambiguous — there is no legitimate reading in which a
   marker carries `CONCERN::` rows and correctly forwards nothing. This is the
   minimum fix and it is squarely the project's fail-fast rule: rc=0 on
   ingest-nothing is a silent default that swallows the fault.
2. **Then decide the format question, once.** Either teach the parser the bare
   `CONCERN::<id> | <SEV> | <text>` row form (it is unambiguous and already
   the shape one reviewer emits), or pin the delimited block in both reviewer
   specs so the two agents agree. Prefer accepting both: the reviewers are
   independent by design, and a parser that tolerates both formats cannot be
   desynced by an edit to one spec. Whichever is chosen, the two specs and the
   parser must state the same thing.
3. **Pin it.** A test asserting that a marker with `CONCERN::` rows and no
   delimited block does not exit 0 silently. That test is the whole point —
   without it this reappears the next time a reviewer spec is edited.

## Acceptance

- Forwarding a marker with `CONCERN::` rows that ingest to zero concerns exits
  non-zero with a message naming the marker and the expected format.
- Both emitted formats forward successfully, or exactly one format is pinned
  in both reviewer specs and the parser matches it.
- A committed test fails if the silent-zero-ingest path is reintroduced.
- Existing `persist_verdict_concerns` behavior on well-formed input is
  unchanged, and `--require-block` keeps its current semantics.

## Provenance

Found from #2387 (cron-wrapper push-timeout bounding) at the round-6 boundary,
while forwarding both reviewers' verdicts to the ledger. The round-6 concerns
were handed to the round-7 implementer by hand as the workaround; the round
itself was not blocked.

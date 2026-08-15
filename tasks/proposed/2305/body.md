---
title: Non-atomic fresh claim-file write in run_claim_queue kills a concurrent worker
  with 'unparseable claim file'
kind: infra
tags:
- claim-file-race
created_at: '2026-08-14T23:27:55Z'
has_clean_result: false
origin_prompt: 'Discovered during /issue 2162 turn-boundary-multipatch recovery relaunch:
  grid worker 1 died on an ''unparseable claim file'' that parsed fine afterwards
  and was owned by a live worker — try_claim''s O_CREAT|O_EXCL fresh path leaves the
  file empty between create and payload write, and a concurrent scanner''s empty read
  is escalated to a hard failure.'
workflow: v1
---
# Non-atomic FRESH claim-file write in `run_claim_queue` kills a concurrent worker with "unparseable claim file"

## Goal

Close a real multi-worker race in the shared claim-file queue used by the issue2094/2162 driver family: a worker scanning a block can read a claim file in the window after it is CREATED but before its payload is WRITTEN, and the empty read is escalated to a hard, run-killing failure.

## The bug

`try_claim` (`scripts/issue2162_run.py:323`) elects the claim winner with `O_CREAT | O_EXCL`:

```python
fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
```

The file therefore EXISTS, zero-length, before the JSON payload is written to `fd`. A second worker reaching the same block inside that window takes the `FileExistsError` branch and does:

```python
try:
    rec = json.loads(path.read_text())
except (json.JSONDecodeError, OSError) as e:
    raise RuntimeError(
        f"unparseable claim file {path} — inconsistent claim state, refusing "
        "to guess (delete it manually after diagnosing the writer)"
    ) from e
```

so a transient empty read becomes a HARD worker death. The docstring's "Atomically claim one block" holds for winner ELECTION but not for content VISIBILITY.

Note the STALE-reclaim path immediately below is already correct — it writes `path.name.tmp.<token>` and `os.replace`s it. The fresh path is the only non-atomic writer, so the fix is to make it match its own sibling.

## Observed instance (#2162, 2026-08-14 23:23Z)

Grid worker 1 died with:

```
RuntimeError: unparseable claim file /workspace/issue2162_out/tbmp/claims/blocks_tbmp/
recency_persona_prompted_d5__tbk3__shuffled.claim — inconsistent claim state,
refusing to guess (delete it manually after diagnosing the writer)
```

Diagnosis that rules out the obvious wrong answer: the named file **parses cleanly** on inspection and is owned by pid 6202 (worker 3) with `ts` at 23:23 — i.e. a LIVE worker had just created it. All five claim files present were individually verified parseable after the death. So this was not a corrupt leftover from a killed process (the first hypothesis, and wrong): it was a read landing inside a live writer's create-then-write window. The failure is transient and leaves no evidence behind, which is exactly why it needs a code fix rather than an operational runbook.

Contention context worth recording, because it explains why a latent race suddenly fired: the queue had 5 remaining blocks and 4 workers scanning them, after a recovery relaunch. Normal operation (45 blocks / 4 workers) makes a same-block collision rare; a nearly-drained queue makes it likely. So the race is expected to bite hardest exactly at end-of-run, and on RE-RUNS/resumes — the highest-value, most-annoying moment to lose a worker.

## Fix options (either is small; prefer the first)

1. **Make the fresh-claim write atomic**, mirroring the stale path: write the payload to `<name>.tmp.<token>` and `os.replace` onto the target, keeping `O_CREAT|O_EXCL` (on the tmp name, or on a separate election sentinel) for winner election so exactly-one-winner is preserved.
2. **Treat an empty/unparseable claim as "in-flight", not fatal**: bounded re-read (a few short randomized sleeps) before raising, so the microsecond window resolves itself and a genuinely corrupt file still fails loudly after the bound.

Keep the hard failure for a claim file that is STILL unparseable after the fix's bound — the fail-fast instinct here is right; only its trigger is too eager.

## Acceptance

- A test that reproduces the window deterministically (create the claim file empty, then have a second worker call `try_claim` on it) no longer raises, and the block is neither double-run nor silently skipped.
- Exactly-one-winner is preserved under concurrent fresh claims (existing behaviour must not regress).
- A genuinely unparseable claim (non-empty garbage that persists) still raises the hard error.
- `release_claim`'s stolen-claim tolerance and the r1 M1 TOCTOU read-back arbitration are unchanged.

## Blast radius

`run_claim_queue` / `try_claim` are shared by the issue2094 + issue2162 driver family (`scripts/issue2162_run.py` is imported as `R` by `scripts/issue2162_tbmp.py`), so any multi-worker run over a nearly-drained queue is exposed, and any resume is exposed from its first scan.

## Provenance

Found by the #2162 orchestrator during the `turn-boundary-multipatch` follow-up round, while relaunching the final 5 grid blocks after the separate HF file-count blocker (#2304 — distinct bug, distinct file, filed separately). Incident detail is in the `epm:progress` notes on #2162 for 2026-08-14.

---
name: sequential-runs-overwrite-shared-summary
description: A driver that subprocess-runs a reused CLI N times into ONE out-root loses all but the LAST run's rows when the CLI's aggregator os.replace-overwrites its summary — trace the summary WRITER's semantics (overwrite vs merge) before approving any multi-invocation shared out-root (#2388 R1 g6)
metadata:
  type: feedback
---

When a new driver composes MULTIPLE sequential invocations of a reused
CLI/entrypoint into the SAME out-root (different budgets/regimes per run —
e.g. a capped leg + legacy companion legs), read the reused code's SUMMARY
WRITER before anything else: an atomic `write_summary(records, path)` that
rebuilds the payload from THIS run's records and `os.replace`s the file
keeps only the LAST invocation's rows, and a per-run resume predicate that
consults only the REQUESTED grid's unit keys does NOT rescue earlier runs'
rows into later summaries. Any downstream aggregation reading that summary
then silently ships a registered anchor missing — and a non-empty guard
(`if not labelled: raise`) cannot fire when the surviving rows share the
label.

**Why:** #2388 R1 g6 (`issue2388_fits.py::phase_h3` stage2): three
`issue1739_fits.py` runs (2500 capped / 8000 / 16000 uncapped) into
`fits/qa`; `arms.write_summary` overwrites `all_arms_spearman.json` per
run, so `h3_parent_exact.json` shipped ONLY the 16,000 legacy rows — the
registered stage-2 kill read (capped 2,500) silently absent. Critical.

**How to apply:** for every multi-invocation shared-out-root pattern,
(1) open the reused writer and classify overwrite-vs-append-vs-merge;
(2) check whether a per-unit append-only sidecar exists (here
`percell/cells.jsonl`, regime-keyed) — aggregating from IT is the cheap
fix; (3) alternatives: per-leg out-roots + explicit merge, or snapshot the
summary between runs; (4) check the aggregation's emptiness guard against
the partial-survival case (rows sharing the filter key defeat it). Sibling
family: [[spend-consumer-accepts-partial-shard-set]] (partial set accepted
as complete), [[inplace-merge-phase-not-idempotent]] (append-side dual).

---
title: 'workflow-fix: upload-verifier reconciles self-reported row counts, not realized
  store contents (passed a 25% shortfall)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-06T08:24:16Z'
has_clean_result: false
origin_prompt: 'Diagnosed on #2091: upload-verifier PASSed while ~25% of activation-capture
  rows were missing; its row-count check read the producer''s _job_done.json capture_rows
  field instead of the realized row_index line counts.'
workflow: v1
---
## Overview / Motivation

The Step 8 `upload-verifier` returned **PASS** on issue #2091 while ~25% of the run's activation-capture rows were missing from the store. The shortfall was found only when a downstream analysis phase crashed on a `KeyError` for an absent context.

The verifier's row-count reconciliation was performed against each job's SELF-REPORTED `_job_done.json` `capture_rows` field, not against the realized row counts inside the store it had just verified the file-presence of.

## Goal

Make per-issue upload-verification reconcile REALIZED artifact contents, so a store that is short INSIDE its files cannot pass.

## What happened (measured, #2091, 2026-08-06)

The verifier reported: "`capture_rows == n_kept == n_contexts` for every job; total 15,970 == the staging manifest's `n_contexts_total`", plus an exact-set reconciliation of 3,130/3,130 files by path AND byte-size, 0 missing, 0 size-mismatched.

Both statements were true and neither caught the problem:

- The **file-level** exact-set check was genuinely exact — every file that EXISTS matches. The store was missing rows INSIDE those files, which a path+size check cannot see.
- The **row-count** check compared `capture_rows` (a producer-written field in `_job_done.json`) against the expected context count. The producer's field said 2000; the realized store held 1,552.

Realized counts, read from `row_index_shard*.jsonl` line counts on HF:

| job | expected | realized rows | distinct ctx | missing |
|---|---|---|---|---|
| wildchat | 2000 | 1552 | 1504 | 496 |
| hal_train | 2000 | 1552 | 1511 | 489 |
| syc_aita | 1304 | 856 | 820 | 484 |
| evil_toxicchat | 671 | 223 | 208 | 463 |

All 9 jobs short by exactly `n_rows - 448`. Underlying producer bug (a resume path treating a 64-row partial pilot shard as a full 512-row shard) is fixed separately on #2091; THIS task is about the verifier not detecting it.

## Proposed change

For any artifact class that carries a per-row index or an explicit row count in its own manifest, the verifier reconciles the REALIZED count — sum the `row_index*.jsonl` line counts (or equivalent realized index) — against the expected count, and FAILs on mismatch. A producer's self-reported `capture_rows`-style field may be reported for context but must never be the quantity the gate is decided on.

Scope note: this is the check the existing exact-set file reconciliation does not and cannot cover. Keep the file-level check; add the realized-content check alongside it.

## Constraints

- **Bounded cost.** Realized-count reconciliation must not require downloading whole tensor stores. Row-index / manifest files are small (KB-scale JSONL); scope the read to those, per the scoped `list_repo_tree` discipline — never a full-store `snapshot_download`.
- **Fail loud, name both numbers.** A mismatch reports expected vs realized per job with the per-shard breakdown, not a bare FAIL.
- **No false FAILs on legitimately-partial classes.** Some artifact classes are legitimately sparse (declared `discarded_artifacts:`, deliberately sampled probes). The check applies where an expected count is declared; state the exemption path.

## Precedent

This is the #1773 lesson recurring at a new site — a per-issue upload-verify must enumerate and reconcile what was REALLY written, never the producer's claim about it. #1773's instance was a missing HF prefix; this one is missing rows inside present files, so the existing prefix-enumeration fix does not cover it.

Worth noting the same error was made independently by the orchestrator during #2091's recovery: a pre-relaunch completeness check also read `capture_rows` from the done records and reported "rows=2000" for a store holding 1,552. Two independent readers made the identical substitution, which suggests the affordance (a convenient self-reported field sitting next to the real data) is the problem rather than either reader.

## Provenance

- workflow_fix_target: .claude/agents/upload-verifier.md
- fingerprint: 8f3c1d20ae74

Diagnosed on task #2091 (2026-08-06). The verifier's row-count reconciliation
read each job's self-reported `_job_done.json` `capture_rows` field instead of
the realized `row_index_shard*.jsonl` line counts, returning PASS while ~25% of
the activation-capture rows were absent from the store. Full measured evidence:
`epm:failure` v3 on #2091.

---
title: 'cron_lesson_consolidate: surface consolidate_lessons.py exit-3 budget refusals'
kind: infra
tags:
- from-2189
created_at: '2026-08-08T02:16:22Z'
has_clean_result: false
origin_prompt: 'task #2189 plan v3 residual 1: cron wrapper''s unconditional exit
  0 swallows the new promote_refused_budget exit code 3; surface it as a notification'
workflow: v1
---
## Overview / Motivation

Filed by task #2189 (plan v3 §11 criterion 7 — the recorded residual gap).

#2189 added a byte-budget guard to `scripts/consolidate_lessons.py::promote()`:
when appending gotcha-candidate bullets to `.claude/rules/gotchas.md` would push
the file past `GOTCHAS_SIZE_WARN_BYTES`, `promote()` refuses (no write), the
refused bullets are printed verbatim, the run summary carries
`promote_refused_budget=<n>`, and `main()` returns **exit code 3** (dedupe/prune
still commit normally).

The nightly cron wrapper `scripts/cron_lesson_consolidate.sh` deliberately ends
with an unconditional `exit 0` (line 54: `# Exit 0 regardless — the log file is
the audit trail, no cron email per routine pass.`). It DOES capture the child rc
in its per-pass log line (line 46: `echo "=== $(date -Iseconds)
lesson_consolidate exit=$rc ==="`), but nothing watches that log — so a budget
refusal (exit 3) is silently swallowed: refused bullets keep re-refusing every
night with no notification until a human happens to read
`logs/lesson_consolidate/*.log`.

## Goal

Surface a `consolidate_lessons.py` exit-3 budget refusal from the nightly cron
boundary as a notification (sidecar row + push, matching the other cron
escalation channels) instead of only a log line, so refused gotcha promotions
are seen the same day rather than silently re-refusing indefinitely.

## Workflow gap

- **Bug observed:** `promote()` budget refusals (exit 3, shipped in #2189) are
  invisible: `cron_lesson_consolidate.sh` logs `exit=$rc` then `exit 0`
  unconditionally, and no watcher/sidecar reads that log.
- **Why it is a workflow gap:** the guard's whole purpose is to force a human
  re-trim decision on gotchas.md; a refusal that nobody sees defers that
  decision forever while the refused lessons stay un-promoted.
- **Confidence (filer):** high
- verified-at-filing: `grep -n "exit" scripts/cron_lesson_consolidate.sh` →
  unconditional `exit 0` at line 54 with the comment "Exit 0 regardless — the
  log file is the audit trail"; child rc captured only into the log at line 46
  (2026-08-07). Exit-3 semantics landed on branch `issue-2189` (commit
  `1d198c04d5`, merging via the #2189 pipeline).

## Proposed change (sketch — refine in planning)

In `scripts/cron_lesson_consolidate.sh`, after the `rc=$?` capture:

```
+ if [ "$rc" -eq 3 ]; then
+     # budget refusal — surface loud (sidecar JSONL row + best-effort push),
+     # dedup one alert per calendar day so a persistent refusal does not spam
+ fi
  exit 0   # routine passes stay silent; cron email stays suppressed
```

Notification channel choice (Telegram push vs `.claude/cache/*-events.jsonl`
sidecar + watcher pass, vs both) is a plan-time decision — mirror whichever
channel the disk-guard tier-(b) escalation uses.

## Constraints / invariants

- Do NOT change `consolidate_lessons.py` exit semantics (exit 3 contract is
  pinned by `tests/test_consolidate_lessons.py` budget cases from #2189).
- Routine (rc=0) passes stay silent — no cron email, `exit 0` retained.
- Do NOT touch the cron schedule or `T_DEDUPE`/`K_RECUR`/`--window-days`.

## Provenance

- Filed from task #2189 implementation round (plan v3 §5.4 "recorded residual
  gap" + §11 criterion 7).

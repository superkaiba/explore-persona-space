---
title: 'cron_lesson_consolidate: an uncreatable log dir silently skips the whole nightly
  pass'
kind: infra
tags:
- from-2190
created_at: '2026-08-08T03:14:29Z'
has_clean_result: false
origin_prompt: 'task #2190 code-review round 1 Minor: unchecked mkdir -p means an
  uncreatable LOG_DIR makes the brace-group redirect fail, so the consolidator never
  runs and the wrapper still exits 0'
workflow: v1
---
# cron_lesson_consolidate: an uncreatable log dir silently skips the whole nightly pass

## Goal

Make `scripts/cron_lesson_consolidate.sh` fail loud when it cannot write its
own daily log file, instead of silently skipping the entire lesson-consolidation
pass and exiting 0.

## The bug

`scripts/cron_lesson_consolidate.sh` runs the consolidator inside a brace group
redirected to its daily log:

```
mkdir -p "$LOG_DIR"
...
{
    echo "=== ... start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/consolidate_lessons.py --apply --window-days 7
    rc=$?
    echo "=== ... exit=$rc ==="
} >> "$LOG_FILE" 2>&1
...
exit 0
```

There is no `set -e`, and `mkdir -p` is unchecked. If `$LOG_DIR` cannot be
created — ENOSPC on `/` (chronically near-full and shared across ~15 concurrent
sessions), a permissions change, a bad `EPS_LESSON_CONSOLIDATE_LOG_DIR`
override — then:

1. `mkdir -p` fails, and nothing notices.
2. The brace group's `>> "$LOG_FILE"` redirect fails, so the **entire group
   never executes** — the consolidator is never invoked at all.
3. `rc` is never assigned.
4. The wrapper reaches `exit 0`.

Net effect: the nightly lesson-consolidation pass silently does not run, and
the only place that would have recorded the fact is the log file that could not
be created. Every marker in the window keeps re-accumulating unprocessed. There
is no cron email on this VM (no MTA; the crontab line redirects `2>&1`), so
nothing else surfaces it either.

Reproduced directly:

```
EPS_LESSON_CONSOLIDATE_LOG_DIR=/proc/definitely_not_writable \
EPS_LESSON_CONSOLIDATE_BIN=/bin/true \
bash scripts/cron_lesson_consolidate.sh
# → wrapper exits 0; consolidator never ran; nothing written anywhere
```

This is the same failure CLASS as #2190 (a real condition that produces no
notification), on the same file, but a DISTINCT condition: #2190 surfaces a
`consolidate_lessons.py` exit-3 budget refusal — a pass that RAN and refused;
this is a pass that never ran at all.

## How it surfaced

Found by the #2190 code-review round-1 pass as a Minor. #2190's diff moved the
`rc` read outside the brace group, which under `set -u` turned this path into
`rc: unbound variable` / exit 1 — an accidental, undiagnostic behaviour change
from a change meant only to ADD an alert path. #2190 restored the pre-diff
behaviour with `${rc:-0}` and deliberately did NOT widen its scope to fix the
underlying silence, filing it here instead.

## Suggested direction (not prescriptive — the planner decides)

The wrapper already has the right precedent a few lines above: the `uv`-on-PATH
check fails loud to stderr and `exit 1`s, with a comment explaining that the
trailing `exit 0` would otherwise hide it (task #580). A `mkdir -p` failure
deserves the same treatment — check it, write a FATAL line to stderr naming the
path, and exit non-zero — plus, given #2190 has now wired a Telegram channel
into this wrapper, consider routing this condition through it too, since stderr
alone reaches nobody under cron.

Whatever shape is chosen must keep #2190's invariants intact: routine rc=0
passes stay silent, the consolidator's exit semantics are untouched, and the
`tests/test_cron_lesson_consolidate.py` pins (T1-T9) stay green.

## Acceptance criteria

- An uncreatable `$LOG_DIR` produces a loud, diagnostic failure naming the path
  — not a silent `exit 0`.
- The failure is pinned by a test in `tests/test_cron_lesson_consolidate.py`
  driving the wrapper with an unwritable log dir.
- All existing T1-T9 pins stay green; rc=0 passes stay silent.
- No change to `scripts/consolidate_lessons.py`, the crontab schedule, or
  `--window-days`.

## Provenance

Surfaced by the #2190 code-reviewer (round 1) as a non-blocking Minor on commit
`d43fc3687c`; #2190 applied the one-token `${rc:-0}` guard (commit `912f4f3a95`)
to avoid shipping an unintended behaviour change, and filed this task for the
underlying condition rather than widening its own approved scope.

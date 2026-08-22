---
title: 'workflow-fix: Step 9c script-file gate launcher — newline-separated test list
  collapses to rc=126 with zero tests collected and evades both anti-silent-pass guards'
kind: infra
tags:
- wf-fix
created_at: '2026-08-15T01:39:35Z'
has_clean_result: false
parent_id: 2302
origin_prompt: hit while running the Step 9c gate during /issue 2302
workflow: v1
---
## Goal

Make the `/issue` Step 9c gate recipe's **script-file variant** state that the substituted test-file
list must be SPACE-separated on ONE line, and give the completion-read a guard that catches the
newline-split failure mode, which today produces zero collected tests and matches none of the
recipe's existing anti-silent-pass greps.

## The gap

SKILL.md Step 9c step 1b writes the gate set into the detached launcher as
`S9C_FILES="<files>"   # verbatim from step 1a's printed command` — a space-separated list. The
`#2115` note directly above says a gate workload needing a script FILE must be composed with the
Write tool rather than a heredoc, but neither passage says what SHAPE the list must have once it
lives in a script.

In a script file the natural way to load 211 paths is a command substitution over a file, which
PRESERVES NEWLINES. Spliced into the recipe's

```
PYTEST_PID=$(bash -c "setsid nohup bash -c 'timeout ... uv run pytest $S9C_FILES --continue-on-collection-errors ...'")
```

the newlines terminate the inner command, so every test path after the first is parsed as its own
COMMAND. Result: `bash: line N: tests/test_x.py: Permission denied` × N, `rc=126`, and **zero tests
collected**.

## Why the existing guards do not catch it

The recipe's two anti-silent-pass guards are (a) the `no tests ran|collected 0 items` FATAL grep and
(b) the missing-rc check. This failure mode evades BOTH:

- pytest never starts, so the log contains neither `no tests ran` nor `collected 0 items` — only
  `Permission denied` lines. Guard (a) does not fire.
- the `; echo $? > rc` unit still runs, so the rc file EXISTS. Guard (b) does not fire.

The gate is not recorded as a false PASS (`rc=126` is non-zero, so it routes to FAIL), so this is a
**diagnosis-cost bug, not a correctness hole** — but the FAIL it produces is unattributable: it looks
like a catastrophic 211-file collapse rather than a one-character list-shape mistake, and the
`#1992`-class "gate set cross-check" that runs BEFORE the launch passes cleanly (the count is right;
only the separator is wrong).

## Evidence

#2302, 2026-08-14: first Step 9c launch (pid 2000225) died exactly this way — 211 `Permission
denied` lines, `rc=126`, no tests collected. Fixed by collapsing the list to one space-separated
line; the relaunch collected normally.

## Proposed fix (direction; the plan decides)

1. **Recipe prose:** in the `#2115` script-file paragraph, state that the file list must be
   space-separated on ONE line and name the newline-preserving-substitution trap explicitly.
2. **A cheap mechanical guard in the launcher preamble**, next to the existing gate-set
   cross-check — assert the substituted list contains no newline, e.g. refuse when
   `printf '%s' "$S9C_FILES" | grep -q '$'`-style multi-line detection fires, or compare
   `wc -l` against `wc -w` expectations. This is strictly cheaper than diagnosing the failure
   after a launch.
3. **Completion-read attribution:** extend the FATAL grep alternation to include the
   `Permission denied` / `rc=126` signature so the verdict names the cause rather than reporting
   an opaque mass failure.

## Acceptance criteria

1. A newline-bearing substituted file list is REFUSED before launch, naming the separator as the
   cause.
2. A correctly space-separated list launches unchanged (no false positives).
3. If the failure does reach the completion-read, the verdict text attributes it to the list shape
   rather than to the tests.
4. The existing gate-set count cross-check and both anti-silent-pass guards keep their current
   behavior.

## Provenance

Hit directly by the #2302 session while running its own Step 9c gate. Filed WITHOUT an auto-spawn
(`file_infra_task.py --no-dispatch`): #2302 is itself a `wf-fix` task, so spawning further fix
sessions off its own incidental findings is the cascade the recursion guard exists to prevent. The
watcher's `proposed_infra_sweep` pass is the documented dispatch backstop. Distinct target and
fingerprint from #2309 (implementer marker four-H3 contract) and from #2302's own payload
(`scripts/select_step9c_tests.py` + `scripts/step9c_baseline.py` classification), so this is not a
dedup hit on either.

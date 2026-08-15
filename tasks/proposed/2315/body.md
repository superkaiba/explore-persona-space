---
title: 'workflow-fix: Step 9c 1b --files-only output needs newline-flattening before
  launcher interpolation; the count cross-check cannot see the un-flattened shape'
kind: infra
tags:
- wf-fix
created_at: '2026-08-15T07:12:29Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2312''s own Step 9c gate round 1: rc=126 with 185x Permission
  denied because --files-only newlines split one pytest command into 186 commands;
  the 186==186 count cross-check passed.'
workflow: v1
---
# Step 9c 1b: `--files-only` output must be newline-flattened before launcher interpolation — the count cross-check cannot see the un-flattened shape

## Goal

Close a gate-launcher composition trap. The Step 9c 1b recipe sanctions
`select_step9c_tests.py --files-only` for "a session composing its own launcher", but does not warn
that its NEWLINE-separated output must be flattened to one line before being interpolated into the
detached `bash -c '…'` launcher string — and the gate-set cross-check the same recipe prescribes
cannot detect the failure.

## Mechanism

`--files-only` emits one test path per line (deliberately — "paths only, one per line, no key to
guess"). Interpolating that value into the quoted inner launcher script preserves the newlines, so
the inner shell parses ONE pytest command as N commands:

```
timeout … uv run pytest tests/test_a.py     <- line 1: runs, reports a small tally
tests/test_b.py                             <- line 2: executed AS A COMMAND -> "Permission denied"
…                                           <- one per remaining file
```

The visible signature is a fast exit with **rc=126** ("found but not executable", from the last
line), a log whose first block is a plausible-looking small PASS tally, and then N−1
`Permission denied` lines. The danger is the tally: a session skimming the log head sees
`5 passed in 0.25s` and a green-looking session start.

## Why the existing 1b cross-check does not catch it

The prescribed guard compares COUNTS:

```bash
S9C_GOT=$(printf '%s\n' $S9C_FILES | grep -c . || true)   # unquoted: word-split is the intent
[ "${S9C_GOT:-0}" -eq "${S9C_N:-0}" ] || { echo "FATAL: ..." >&2; exit 1; }
```

Word-splitting treats spaces and newlines identically, so a correctly-sized but newline-separated
list yields the right count and passes. The check was designed against the #1992 shape (an EMPTY
list, which makes a bare `pytest` collect the whole suite) — a real and different failure. It is
count-complete and separator-blind.

## Measured incident (#2312, 2026-08-15)

Round 1 of #2312's Step 9c gate:

```
5 passed in 0.25s
bash: line 2: tests/test_adversarial_planner_factchecker_grain_pin.py: Permission denied
...  (185 total)
rc=126
```

186 files selected; the cross-check printed `gate set cross-checked: 186 test files (expected 186)`
and passed. Cost: one wasted launch cycle plus the diagnosis; recovered by flattening with
`| tr '\n' ' '` and adding an embedded-newline assert. No compute was lost (0.25 s of test time), but
the same shape on a longer-running gate would waste a full cycle, and a session that read only the
log head could mistake the partial tally for a result.

## Proposed fix (direction only — the plan decides)

1. In the Step 9c 1b recipe, where `--files-only` is named as the self-composed-launcher form, state
   that the output is NEWLINE-separated and MUST be flattened before interpolation
   (`--files-only | tr '\n' ' '`), with the reason (the inner shell splits on newlines).
2. Extend the gate-set cross-check with a separator assert alongside the count assert — the
   substituted set must carry ZERO embedded newlines — so the shape fails CLOSED pre-launch instead
   of after a wasted launch.
3. Consider whether the same trap reaches the other sites that interpolate a selected file list into
   a launcher string (the Step 10d lint-gate blocks, the Step 9a-ter inline payload lint gate) and
   fix them in the same family if so.
4. Consider adding `rc=126` + `Permission denied` to the documented gate failure signatures, so the
   next session recognizes it immediately rather than diagnosing from scratch.

## Acceptance

1. The 1b recipe names the flattening requirement at the point where `--files-only` is sanctioned.
2. A newline-separated substituted set fails the pre-launch cross-check with a legible message.
3. A pin test reproduces the un-flattened shape and asserts the new guard refuses it.
4. The ordinary (already-flattened) launcher path is behaviourally unchanged.

## Provenance

Surfaced by the #2312 orchestrator during its own Step 9c gate (round 1, rc=126). Filed `proposed`
and deliberately NOT auto-spawned: #2312 is itself a `wf-fix` task (recursion-guard intent), and a
spawned sibling would edit `.claude/skills/issue/SKILL.md` while #2312 has that file in flight —
reproducing the concurrent-same-file collision #2296, #2302 and #2312 have each already paid for.
Dispatch after #2312 lands.

---
title: 'workflow-fix: Step 9c gate launcher file-list cross-check is newline-agnostic
  — split-argv launch reports a plausible PYTEST_RC that is not a verdict'
kind: infra
tags:
- wf-fix
created_at: '2026-08-14T15:49:50Z'
has_clean_result: false
origin_prompt: 'Surfaced during /issue 2293 Step 9c: --files-only newline-separated
  list interpolated into the inner bash -c split the launch into per-path commands;
  rc=126 with ''5 passed in 0.42s'' looked like a verdict; the recipe''s word-splitting
  count check reported 151==151 and could not catch it.'
workflow: v1
---
# `/issue` Step 9c gate launcher: the file-list cross-check is newline-agnostic, so a split-argv launch reports a plausible `PYTEST_RC` that is not a verdict

## Goal

Make `.claude/skills/issue/SKILL.md`'s Step 9c step 1b (and the step 1c/1d twins that
inherit the same substitution shape) structurally unable to launch the gate with a
**newline-separated** test-file list. Today the mandated cross-check validates only the
COUNT, so a newline-bearing list passes the check, prints `gate set cross-checked`, and
then silently degenerates into a one-file run whose `PYTEST_RC` looks like a real gate
result.

## Observed incident (#2293, 2026-08-14)

Step 9c for #2293 (151 selected files) was launched with the file list derived
programmatically:

```
S9C_FILES=$(uv run python scripts/select_step9c_tests.py --files-only)
```

`--files-only` emits one path per line. Interpolated into the inner `bash -c` script,
those newlines survived into the inner shell, which parsed:

- **line 1** as `timeout ... uv run pytest tests/test_adhoc_summary_disclosure_pins.py`
  — the ONLY file that ran. It produced a genuine-looking `5 passed in 0.42s`.
- **lines 2..151** as their own COMMANDS, each trying to execute a test file as a
  program: 150 x `bash: line N: tests/<name>.py: Permission denied`.
- **the trailing flag line** as yet another command, so `--junitxml`,
  `--continue-on-collection-errors`, `-o junit_family=xunit1` and `--basetemp` NEVER
  reached pytest.

`PYTEST_RC=126` ("command found but cannot execute") was the exit of the last bogus
command. Two properties make this dangerous rather than merely broken:

1. **The failure is disguised as a verdict.** `rc=126` plus a log tail reading
   `5 passed in 0.42s` is readable as a real gate outcome. Only the absence of
   `/tmp/step9c-junit-issue-<N>.xml` — a file the recipe's own step 1d then consumes —
   reveals that 150/151 files never ran.
2. **The mandated cross-check cannot catch it.** The recipe's guard is:

   ```
   S9C_GOT=$(printf '%s\n' $S9C_FILES | grep -c .)   # unquoted: word-split is the intent
   [ "$S9C_GOT" -eq "$S9C_N" ] || FATAL
   ```

   Word-splitting treats newlines and spaces identically, so this reported `151 == 151`
   and printed `[step9c] gate set cross-checked: 151 test files` while the launch was
   already broken. The check is structurally blind to this class — it validates COUNT,
   never single-line-ness.

## Root cause

The recipe carries an **unstated** assumption. `S9C_FILES="<files>"   # verbatim from
step 1a's printed command` assumes a human-pasted, space-separated single line, and the
count check exists to catch an unsubstituted placeholder or a truncated paste. Deriving
the list programmatically — which is strictly safer against transcription error, and is
the natural thing for an orchestrator to do — violates that assumption silently. Nothing
in the recipe states the list must be single-line, and nothing checks it.

## Proposed fix

1. **Normalize at the source.** In step 1b (and any sibling that builds a file argv),
   mandate the collapse explicitly:
   `S9C_FILES=$(uv run python scripts/select_step9c_tests.py --files-only | tr '\n' ' ')`
   with a comment saying `--files-only` is newline-separated and why raw interpolation
   breaks the inner `bash -c`.
2. **Add a newline assertion beside the count check**, since the count check cannot
   cover it:
   ```
   S9C_NL=$(printf '%s' "$S9C_FILES" | tr -cd '\n' | wc -c)
   [ "$S9C_NL" -eq 0 ] || { echo "FATAL: file list holds $S9C_NL newline(s) — the inner bash would split it into per-path commands" >&2; exit 1; }
   ```
   **Do NOT** write the guard as `case "$S9C_FILES" in *"$(printf '\n')"*)` — command
   substitution strips trailing newlines, so that pattern degenerates to `*""*` and
   matches EVERY string, firing unconditionally. This was tried during #2293's recovery
   and aborted a healthy launch. Worth an inline comment: the always-true form is an easy
   and silent mistake.
3. **Post-launch recurrence probe** (cheap, catches the class even if a future variant
   dodges 1+2): after the launcher, assert the log has no `Permission denied` line and
   that the argv on `ps` shows more than one path. #2293's attempt 3 did exactly this and
   confirmed the fix engaged.
4. **Consider a junit-presence precondition in step 1d.** Compare consumes
   `/tmp/step9c-junit-issue-<N>.xml`; a missing junit currently surfaces only as a
   downstream compare failure. An explicit "junit absent ⇒ 1b did not really run, apply
   1b's FAIL path" check would have caught the incident immediately.

## Acceptance

1. The Step 9c 1b recipe in `.claude/skills/issue/SKILL.md` mandates the newline collapse
   AND carries the newline-count assertion, with the always-true `case`/`$(printf '\n')`
   form called out as the wrong way to write it.
2. A pin test asserts the recipe text contains both the collapse and the assertion (the
   `tests/test_issue_skill_gate_*` family is the natural home — e.g. alongside
   `test_issue_skill_gate_recipe_hardening.py`).
3. The count-only cross-check's blind spot is stated in the recipe prose, so a future
   reader does not re-derive the same false confidence from a passing
   `gate set cross-checked` line.
4. No change to the gate's selection, timeout derivation, detachment shape, choom
   handling, or compare semantics.

## Provenance

Surfaced by #2293's own Step 9c gate (task #2293 `epm:progress`, 2026-08-14 — the
`[long-phase-heartbeat] Step 9c gate RELAUNCHED` note carries the full three-attempt
forensics). #2293's subject matter is unrelated (it fixes the pristine-oracle base sha in
`scripts/step9c_baseline.py`); this task is the SKILL.md launcher gap that its gate run
exposed. Distinct target file, distinct fingerprint.

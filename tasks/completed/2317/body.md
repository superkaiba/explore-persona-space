---
title: 'workflow-fix: Step 9c gate-set cross-check validates cardinality but not splice
  shape — a newline-separated --files-only list runs 1 of N files behind a green guard'
kind: infra
tags:
- step9c-splice-shape
created_at: '2026-08-15T08:16:01Z'
has_clean_result: false
origin_prompt: 'hit in #2314''s Step 9c launch: a --files-only-sourced gate set spliced
  newlines into the launcher command text, so pytest ran 1 of 120 files, no junit
  was written, and rc=126 landed in the sentinel — while the canonical count cross-check
  printed ''gate set cross-checked: 120 test files'''
workflow: v1
---
---
kind: infra
---

# workflow-fix: the Step 9c gate-set cross-check validates CARDINALITY but not SPLICE SHAPE, so a newline-separated `--files-only` list launches a gate that silently runs 1 of N files

## Goal

Harden the canonical Step 9c gate-launch recipe in `.claude/skills/issue/SKILL.md`
so that a gate set sourced from `select_step9c_tests.py --files-only` (one path per
LINE) cannot be spliced into the launcher's command text as multiple commands — and
so the existing cross-check FAILS LOUD on that shape instead of passing.

## The gap

The canonical launch block (SKILL.md Step 9c 1b, the `#1992`/`#2126` gate-set
cross-check) guards the gate set like this:

```sh
S9C_N=$(uv run python scripts/select_step9c_tests.py --files-only 2>/dev/null | grep -c .)
S9C_FILES="<files>"   # verbatim from step 1a's printed command
S9C_GOT=$(printf '%s\n' $S9C_FILES | grep -c . || true)   # unquoted: word-split is the intent
[ "${S9C_GOT:-0}" -eq "${S9C_N:-0}" ] || { echo "FATAL: ..."; exit 1; }
```

`$S9C_FILES` is then spliced into the command TEXT of the detached launcher:

```sh
PYTEST_PID=$(bash -c "setsid nohup bash -c 'timeout ... uv run pytest <files> ...' ...")
```

The check counts paths after **unquoted word-splitting**, and `IFS` includes
newline — so a NEWLINE-separated list counts correctly (120 == 120, check passes)
while splicing incorrectly: the newlines are expanded into the command string, and
the inner `bash -c` parses every path after the first as its own COMMAND.

The guard therefore validates the gate set's CARDINALITY, never its SPLICE SHAPE.
The placeholder comment (`verbatim from step 1a's printed command`) implies a
space-separated literal, but nothing enforces it, and `--files-only` — whose entire
purpose is programmatic consumption, and which the cross-check itself invokes one
line earlier — emits the newline shape.

## Observed failure (#2314, 2026-08-15)

A gate launched with a `--files-only`-sourced list produced:

- pytest ran **1 of 120** selected files (`test_adhoc_summary_disclosure_pins.py`,
  5 passed); every remaining path failed as a bogus command
  (`bash: line 2: tests/...: Permission denied`, one per file).
- The trailing flags (`--continue-on-collection-errors`, `--junitxml=...`,
  `--basetemp=...`) attached to the LAST bogus line, so the single real pytest
  invocation ran WITHOUT `--junitxml` — **no junit XML was written at all**.
- `rc=126` (shell "command found but cannot be invoked") landed in the rc
  sentinel — a value with no relationship to any test outcome.
- The cross-check printed `gate set cross-checked: 120 test files` immediately
  before the launch.

Severity is bounded but the shape is nasty: 1/120 coverage behind a green-looking
guard, and an rc that a reader can mistake for a test verdict. The missing junit is
what makes it self-limiting today — a downstream `compare` step has nothing to
parse, so the run cannot be silently accepted as PASS through the junit path. That
is luck, not design: the protection comes from an artifact's ABSENCE rather than
from a check.

## Proposed fix (small; the implementer should keep it minimal)

1. In the canonical block, source the list in the splice-safe shape — pipe
   `--files-only` through `| tr '\n' ' '` — and say in the comment WHY (a
   newline-bearing list parses as separate commands).
2. Add a SPLICE-SHAPE assertion beside the existing count check:
   `[ "$(printf '%s' "$S9C_FILES" | wc -l)" -eq 0 ]`, FATAL otherwise. Prefer
   `wc -l` over a `case`/glob newline test: the obvious-looking
   `case "$S9C_FILES" in *"$(printf '\n')"*)` is BROKEN — command substitution
   strips trailing newlines, so the pattern degenerates to `*""*` and matches
   every string, firing unconditionally. (That false-positive was hit in #2314's
   second launch attempt; it failed safe, but a guard that cannot fail correctly
   is not a guard.)
3. Consider having the launcher SELF-TEST the shape assertion before trusting it
   (a space list must score 0, a newline list must score 1) — three lines, and it
   is what caught the broken guard in #2314.
4. Apply to EVERY hooked gate site that splices a file list, not just 9c 1b — the
   same block is referenced by 9c 1c/1d, both Step 10d gate blocks, and the Step
   9a-ter inline payload lint gate. Grep for the sibling copies of the
   cross-check; SKILL.md carries several near-duplicates of this paragraph.

## Acceptance criteria

1. A regression test asserting the recipe text in SKILL.md carries BOTH the
   `tr '\n' ' '` normalization (or equivalent) AND a newline/splice-shape
   assertion at the gate-launch site — the existing
   `tests/test_issue_skill_gate_recipe_hardening.py` is the natural home.
2. Every sibling copy of the cross-check paragraph in SKILL.md is updated
   consistently (enumerate them in the report; a partial fix leaves the trap live
   at the un-updated sites — the #734 class-sweep discipline).
3. State explicitly whether `select_step9c_tests.py --files-only` should instead
   grow a `--files-only --sep=space` (or NUL-delimited) mode; if the fix stays
   recipe-side only, say why.
4. The no-flags `workflow_lint.py` bundle and the Step 9c selector/baseline test
   suites stay green.

## Scope note

The proximate cause in #2314 was an operator deviation from the documented
`<files>` placeholder, and this task should not pretend otherwise. It is worth
fixing anyway because the deviation is the NATURAL one (the recipe itself calls
`--files-only` one line above the splice), the guard that exists reports success
on it, and the blast radius is a gate that certifies 1/N of its selection.

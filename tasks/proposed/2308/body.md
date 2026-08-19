---
title: 'Step 9c 1b launcher: --files-only list spliced into the bash -c string runs
  one command per path (rc=126); gate-set cross-check is blind to newlines'
kind: infra
tags: []
created_at: '2026-08-15T00:57:05Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2303 orchestrator while running its own Step 9c gate:
  the documented --files-only path produced a launcher that exited 126 with one Permission
  denied per test file, and the 1b gate-set cross-check passed anyway because unquoted
  word-splitting treats a newline as whitespace.'
workflow: v1
---
# Step 9c 1b launcher: a `--files-only` list spliced into the `bash -c` string parses one command PER PATH (rc=126), and the gate-set cross-check is structurally blind to it

`kind: infra`. Workflow-surface gap in `.claude/skills/issue/SKILL.md` § Step 9c step 1b (the detached gate launcher recipe). Observed live on #2303, 2026-08-14. Distinct from #2303's own scope (Step 5a family arm + the Step 10d twin) — same file, different sub-block, so this is its own task rather than a #2303 round.

## The defect

Step 9c step 1a tells a session composing its own launcher to use `--files-only`:

> a session composing its own launcher uses `--files-only` (paths only, one per line — no key to guess) and refuses on empty per the 1b gate-set cross-check.

Step 1b then substitutes that list into the detached launcher, inside a **double-quoted** `bash -c "..."` string:

```
S9C_FILES="<files>"   # verbatim from step 1a's printed command
...
PYTEST_PID=$(bash -c "setsid nohup bash -c 'timeout ... uv run pytest $S9C_FILES ... ' ...")
```

`--files-only` emits **one path per line**. A newline inside that double-quoted string is a **command separator**, not an argument separator. So the launcher does not run one pytest over 191 files — it tries to execute 191 separate commands, each a test-file path. Every one is non-executable, so the shell emits `Permission denied` per path and the wrapper exits **126**. pytest never starts and no junit is written.

The recipe is self-consistent only if `<files>` is taken from step 1a's **printed command** (space-separated on one line). The `--files-only` affordance — offered in the same breath, and the safer-looking option because there is "no key to guess" — silently produces the broken shape.

## Why the existing guard does not catch it

The 1b gate-set cross-check exists precisely to make a mis-copied `<files>` loud:

```
S9C_GOT=$(printf '%s\n' $S9C_FILES | grep -c . || true)   # unquoted: word-split is the intent
[ "${S9C_GOT:-0}" -eq "${S9C_N:-0}" ] || { echo "FATAL: ..." ; exit 1; }
```

The expansion is **unquoted**, so bash word-splits on `$IFS` — which contains newline as well as space and tab. A newline-separated list therefore counts **identically** to a space-separated one. On #2303 the check reported `191` against a selector count of `191`, printed `gate set cross-checked: 191 test files`, and passed. The guard cannot distinguish the two shapes by construction.

## Observed failure signature (for the fix's regression test)

- `rc=126` in `/tmp/step9c-rc-issue-<N>`
- `/tmp/step9c-pytest-issue-<N>.log` = N lines of `bash: line <k>: tests/test_<name>.py: Permission denied`
- **no** `/tmp/step9c-junit-issue-<N>.xml`
- the single-flight probe goes CLEAR within ~0 s of launch, so a probe-keyed Monitor fires an immediate false-DONE

That last point is the compounding hazard: the false-DONE is indistinguishable from a legitimately fast gate unless the session applies verify-first discipline. On #2303 it was caught only because a 0-second completion for a 191-file run is obviously impossible. A less suspicious operator — or a smaller selection — could read the CLEAR probe as success and proceed to a compare against a junit that does not exist.

## Proposed fix (implementer picks; all three are cheap and compose)

1. **Join at the source.** In the 1b recipe, build the variable as `S9C_FILES=$(tr '\n' ' ' < <list-file>)` — or have `--files-only` gain a `--sep` / one-line mode — so the documented path cannot produce the broken shape.
2. **Make the cross-check newline-aware.** Add a guard next to the count check that fails loud on an embedded newline, e.g. `case "$S9C_FILES" in *$'\n'*) echo "FATAL: file list contains newlines — join with spaces before splicing into the bash -c string" >&2; exit 1 ;; esac`. This is the one that would actually have caught it, and it is two lines.
3. **Post-launch liveness assert.** After the identity verify, confirm a `python3?` child exists in the detached session (`pgrep -s "$PYTEST_PID"`) and that the log's first lines are a pytest session banner rather than `Permission denied`, before printing the `gate detached` breadcrumb. Turns a dead launch into an immediate loud failure instead of a false-DONE 45 minutes later.

## Acceptance criteria

1. The documented 1b path, followed literally end-to-end with `--files-only`, produces a launcher that actually runs pytest.
2. A newline-bearing `S9C_FILES` fails loud BEFORE launch, with a message naming the join remedy.
3. A pin test reproduces the rc=126 shape against the pre-fix recipe text and passes post-fix — `tests/test_issue_skill_gate_recipe_hardening.py` or `tests/test_issue_skill_step9c_compare_background.py` are the natural homes.
4. No change to the single-flight probe semantics, the gate-fleet arbitration, or the detached-launcher `setsid`/`$!` shape (the #2005/#1893 contracts).

## Provenance

Surfaced by the #2303 orchestrator session while running its own Step 9c gate, 2026-08-14. First launch pid 1407692 died rc=126; relaunch pid 1448896 with the list space-joined ran correctly. #2303 applied remedies 1 and 2 locally in its own launch command only — nothing in the repo was changed for this, so the recipe in SKILL.md still carries the defect.

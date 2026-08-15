---
title: 'Step 9c 1b: a self-composed gate launcher can silently drop --files-only newline
  flattening OR -o junit_family=xunit1; neither omission fails where the recipe checks'
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
# Step 9c 1b: a self-composed gate launcher can silently drop TWO required elements — `--files-only` newline flattening and `-o junit_family=xunit1`; neither failure is visible where the recipe checks

## Goal

Close a gate-launcher composition trap with two demonstrated instances. The Step 9c 1b recipe
sanctions `select_step9c_tests.py --files-only` for "a session composing its own launcher", but a
self-composed launcher can silently omit either of two elements the canonical launcher line carries.
Neither omission surfaces where the recipe tells the session to look: one produces a
plausible-looking partial PASS, the other produces an INDETERMINATE compare that reads like a tool
problem rather than a launcher defect.

## Instance A — `--files-only` output is NEWLINE-separated

`--files-only` emits one test path per line (deliberately — "paths only, one per line, no key to
guess"). Interpolating that value into the quoted inner `bash -c` launcher preserves the newlines,
so the inner shell parses ONE pytest command as N commands:

```
timeout … uv run pytest tests/test_a.py     <- line 1: runs, reports a small tally
tests/test_b.py                             <- line 2: executed AS A COMMAND -> "Permission denied"
…                                           <- one per remaining file
```

Visible signature: a fast exit with **rc=126** (from the last line), a log whose first block is a
plausible small PASS tally, then N−1 `Permission denied` lines. The danger is the tally — a session
skimming the log head sees `5 passed in 0.25s` and a green-looking start.

**Why the prescribed 1b cross-check misses it.** The guard compares COUNTS:

```bash
S9C_GOT=$(printf '%s\n' $S9C_FILES | grep -c . || true)   # unquoted: word-split is the intent
[ "${S9C_GOT:-0}" -eq "${S9C_N:-0}" ] || { echo "FATAL: ..." >&2; exit 1; }
```

Word-splitting treats spaces and newlines identically, so a correctly-sized but newline-separated
list yields the right count and passes. The check was designed against the #1992 shape (an EMPTY
list, which makes a bare `pytest` collect the whole suite) — a real and different failure. It is
count-complete and separator-blind.

## Instance B — a self-composed launcher can omit `-o junit_family=xunit1`

The canonical launcher line (SKILL.md ~L10242 / ~L10377) carries `-o junit_family=xunit1` alongside
`--junitxml=`. It is load-bearing for the NEXT step, not for pytest: `step9c_baseline.py compare`
resolves each failing testcase to a repo path via the junit `file` attribute, which only the xunit1
family emits. Omit the flag and pytest still writes a junit and still exits with the right rc — the
gate looks completely healthy — but the compare aborts:

```
step9c_baseline: failing testcase tests.test_shared_vm_thread_caps::test_no_new_torch_before_dotenv_vm_entrypoints
  has no file attribute — xunit1 contract violated (see plan #1022 K2 fallback)
compare rc=2 → indeterminate: True, pristine_oracle: None
```

The failure mode is the misleading part: `indeterminate` is exactly what the compare also returns
for legitimate tool-side reasons (a missing `--run-pristine`, a dirty-root oracle it cannot
neutralize), so the natural reading is "the baseline tool could not decide" rather than "my
launcher was malformed 30 minutes ago". The tests must be re-run to recover a usable verdict; there
is no way to reconstruct the `file` attributes after the fact.

## Measured incidents (both in #2312, 2026-08-15)

**A** — gate round 1: 186 files selected, the cross-check printed
`gate set cross-checked: 186 test files (expected 186)` and PASSED, and the run exited rc=126 in
0.25 s with 185 × `Permission denied`. Recovered by flattening with `| tr '\n' ' '` plus an
embedded-newline assert.

**B** — the post-merge re-gate (after sibling #2303 landed on the same files mid-gate): two
hand-composed part launchers both omitted `-o junit_family=xunit1`. Part 1 ran to completion
(**1 failed, 2348 passed in 1787.80 s**) and its compare then returned `indeterminate: True` with
`pristine_oracle: None`. Recovered by re-running the single failing FILE with the flag (25.82 s) and
comparing that junit — `rc=0`, `new: []`, the failure correctly stripped via `pristine-scratch`.
Cheap here only because exactly one file had failed; with failures spread across many files the
recovery is a full re-run.

Same root cause in both: the recipe invites COMPOSING a launcher, and the elements that matter are
easy to drop because nothing between the drop and the damage checks for them.

## Proposed fix (direction only — the plan decides)

1. At the point where `--files-only` is sanctioned, state that the output is NEWLINE-separated and
   MUST be flattened before interpolation (`--files-only | tr '\n' ' '`), with the reason.
2. Extend the gate-set cross-check with a separator assert alongside the count assert — the
   substituted set must carry ZERO embedded newlines — so instance A fails CLOSED pre-launch.
3. Make the `--files-only` sanction point at the FULL canonical launcher line rather than leaving
   the session to reassemble it, so `-o junit_family=xunit1` cannot be dropped by omission. Consider
   an even stronger form: have `select_step9c_tests.py` emit the whole ready-to-run launcher command
   (a `--emit-launcher` mode), making the self-composition path unnecessary.
4. Consider a pre-launch assert that the composed pytest argv contains `junit_family=xunit1`
   whenever it contains `--junitxml` — the cheapest possible closure of instance B, and it also
   protects the Step 9a-ter inline payload lint gate and the Step 10d lint blocks, which interpolate
   selected file lists the same way.
5. Consider adding `rc=126` + `Permission denied`, and the `has no file attribute` compare abort, to
   the documented gate failure signatures so the next session recognizes both immediately.

## Acceptance

1. The 1b recipe names the flattening requirement at the point where `--files-only` is sanctioned.
2. A newline-separated substituted set fails the pre-launch cross-check with a legible message.
3. A composed argv carrying `--junitxml` without `junit_family=xunit1` is refused (or made
   impossible) before launch.
4. Pin tests reproduce both shapes and assert the new guards refuse them.
5. The ordinary (correctly-composed) launcher path is behaviourally unchanged.

## Provenance

Both instances surfaced in the #2312 orchestrator's own Step 9c gate — A in round 1, B in the
post-merge re-gate. Filed `proposed` and deliberately NOT auto-spawned: #2312 is itself a `wf-fix`
task (recursion-guard intent), and a spawned sibling would edit `.claude/skills/issue/SKILL.md`
while #2312 has that file in flight — reproducing the concurrent-same-file collision #2296, #2302,
#2303 and #2312 have each already paid for. Dispatch after #2312 lands.

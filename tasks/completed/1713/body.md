---
title: 'daily-fix: restore check_inline_round_duty_mirror — main red'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1007e9704d5f
- daily-auto-filed
- trigger-dense
created_at: '2026-07-27T07:13:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): check_inline_round_duty_mirror
  was added to workflow_lint.py by #1701 (10eda16ed3) and removed 45 min later by
  #1698''s merge (a5c9a09427), a branch snapshot predating #1701; the test importing
  it survived, so full-suite pytest collection aborts rc=2 fleet-wide'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 2 independent
miner group(s) over the 2026-07-26 session transcripts.

**URGENT — `main` is red RIGHT NOW and has been for ~15.5 h.** Every full-suite `pytest`
run on `main` aborts at collection with rc=2, and `scripts/step9c_baseline.py` explicitly
refuses to classify a rc=2 run (`"refusing to classify a partial run (MF-1b)"`,
`step9c_baseline.py:1826`). The Step 9c test-verdict gate is therefore unusable
fleet-wide without a hand-applied `--ignore`/grep exclusion: every session that reaches
Step 9c between now and the fix pays a diagnosis-plus-workaround tax, and any session
that does not notice ships past a gate that never classified anything.

## Goal

Restore `check_inline_round_duty_mirror` to `scripts/workflow_lint.py` so the full suite
collects again, and close the class of silent lost-update that removed it — a long-lived
branch's whole-file snapshot of a shared file reverting a sibling's merged work between
branch-cut and merge.

## Workflow gap

- **Bug observed:** `tests/test_workflow_lint_inline_round_duty_mirror.py:39` imports
  `check_inline_round_duty_mirror` from `scripts/workflow_lint.py`, but the symbol has
  been absent from `main` since 15:03Z on 2026-07-26, so every collection of the full
  suite dies with `ImportError` and rc=2.
- **Why it is a workflow gap:** the Step 10d merge gate has no check that a branch's copy
  of a shared workflow-surface file does not DELETE lines another commit added after the
  branch's merge-base, so a whole-file branch snapshot silently reverts a sibling's
  merged work with no gate, no conflict, and no warning.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c 'check_inline_round_duty_mirror' scripts/workflow_lint.py`
  → **0 hits** (absence-of-symbol claim; the 0-hit IS the evidence);
  `grep -n 'check_inline_round_duty_mirror' tests/test_workflow_lint_inline_round_duty_mirror.py`
  → **5 hits**, incl. `39:from workflow_lint import check_inline_round_duty_mirror  # noqa: E402`;
  repo-wide relocation grep
  `grep -rln 'check_inline_round_duty_mirror' --include='*.py' --include='*.md' .` (logs/ excluded)
  → **3 files**: `tests/test_workflow_lint_inline_round_duty_mirror.py`,
  `tasks/completed/1701/plans/v2.md`, `tasks/completed/1701/plans/v3.md` — i.e. the symbol
  was NOT relocated, it was deleted;
  `uv run pytest --collect-only -q` → **rc=2**,
  `16502/16521 tests collected (19 deselected), 1 error`,
  `ERROR tests/test_workflow_lint_inline_round_duty_mirror.py` /
  `E   ImportError: cannot import name 'check_inline_round_duty_mirror' from 'workflow_lint'`;
  `grep -n 'refusing to classify' scripts/step9c_baseline.py` → **1 hit at L1826**.
  Commit SHAs resolved: `git rev-parse --verify --quiet '10eda16ed3^{commit}'` →
  `10eda16ed3bb56fe09dc0d4193c8173337e4b5ec`;
  `git rev-parse --verify --quiet 'a5c9a09427^{commit}'` →
  `a5c9a09427b1c20f0a0ab7bd02097450a272301c`. (2026-07-26)

## Evidence

- Verified at compose time: `10eda16ed3` (`workflow-fix #1701: inline-round
  estimator-validity + record-integrity duties`, 2026-07-26T07:18:00-07:00 = 14:18Z) added
  BOTH the `check_inline_round_duty_mirror` function and its test file.
  `a5c9a09427` (`issue-1698: workflow-fix — RunPod launch-path branch/teardown contract`,
  2026-07-26T08:03:21-07:00 = 15:03Z) removed the function 45 min later:
  `git diff --numstat 10eda16ed3 a5c9a09427 -- scripts/workflow_lint.py` → `17  153`, and
  `git log -S'check_inline_round_duty_mirror' --oneline -- scripts/workflow_lint.py`
  returns exactly those two commits (add, then remove). #1698 never touched the test file,
  so the test survived on `main` importing a symbol that no longer exists.
- The removal is a lost update, not a deliberate revert: #1698's branch carried a snapshot
  of `workflow_lint.py` that predated #1701, and the merge applied the branch side
  wholesale. Session `e3b70618` independently discovered the same shape from the other
  direction at 16:10Z, when its own Step-10d spec-freshness sync would have RE-ADDED the
  139 lines: `"Dropped stale spec-freshness sync commit (would have reverted #1698's
  deliberate removal of check-inline-round-duty-mirror)."` — that session read the removal
  as deliberate and dropped its (correct) restoration.
- Session `a2c4bae3`, 2026-07-26T15:52:47Z: hit the red at its Step 9c gate and burned
  3 aborted gate launches plus ~27 min of diagnosis (15:52 → 16:19) before excluding the
  file by hand. Its gate output: `COMPARE_RC=2 … "reason": "pytest rc 2
  (aborted/interrupted/internal-error run) — refusing to classify a partial run (MF-1b)"`.
- Session `a2c4bae3`, 2026-07-26T16:57:18Z: diagnosed the breakage correctly and wrote it
  into its `epm:test-verdict` and `epm:done` notes — `"**Pre-existing red on main (for
  /daily follow-up):** \`tests/test_workflow_lint_inline_round_duty_mirror.py\` imports
  \`check_inline_round_duty_mirror\` from \`scripts/workflow_lint.py\`, but the symbol has
  0 hits on \`origin/main:scripts/wo…"` — but did NOT call `scripts/file_infra_task.py`
  and did NOT emit a `<!-- workflow-fix-candidate v1 -->` block carrying the #1681 urgent
  grammar (`urgency: main-red` + `failing_test:` + `wf_fix:`), which is the only form the
  watcher's `urgent_wf_park_pass` can route within a tick. A prose "noted for /daily
  follow-up" was the terminal disposition.
- Measured cost: ~15.5 h of live main-red (15:03Z 2026-07-26 → this sweep), still live at
  compose time; ~27 min of direct diagnosis in `a2c4bae3` plus 3 aborted gate launches;
  #1701's entire shipped lint check is silently absent from `main`; every session touching
  full-suite scope since 15:03Z pays the same tax.

## Proposed change

- **(a) Restore.** Re-apply the 153 deleted lines to `scripts/workflow_lint.py` from
  `git show 10eda16ed3:scripts/workflow_lint.py` — the `check_inline_round_duty_mirror`
  function plus its registration in the no-flags default bundle. Re-verify with
  `uv run pytest tests/test_workflow_lint_inline_round_duty_mirror.py` (green) and
  `uv run pytest --collect-only -q` (rc=0). Reconcile against #1698's own intended change
  to `workflow_lint.py`: the restoration must ADD #1701's lines back without reverting
  #1698's `17` added lines. `git diff 10eda16ed3 a5c9a09427 -- scripts/workflow_lint.py`
  is the exact reconciliation surface.
- **(b) Close the class — a lost-update guard at the Step 10d merge gate.** In
  `.claude/skills/issue/SKILL.md` Step 10d (the guards / re-snapshot block), add a check
  that, for each shared workflow-surface file the branch touches, diffs the branch copy
  against `origin/main` at merge time and REFUSES the merge when the branch side DELETES
  lines that landed on `origin/main` AFTER the branch's merge-base. Mechanically:
  `git merge-base HEAD origin/main` → for each touched shared path, if
  `git diff --numstat <mb> origin/main -- <path>` shows additions AND
  `git diff --numstat HEAD origin/main -- <path>` shows those same lines present only on
  `origin/main`, the branch is carrying a stale whole-file snapshot — block and name the
  path.
- **(c) Pin it.** Add a regression test under `tests/test_workflow_lint.py` (or a sibling
  gate-contract pin) asserting `check_inline_round_duty_mirror` is importable from
  `workflow_lint` and is registered in the no-flags default bundle, so the symbol cannot
  silently vanish again without a red test that is itself collectible.
- **(d) Route the diagnosis next time.** In `.claude/skills/issue/SKILL.md` Step 9c, make
  the pre-existing-red exclusion path MANDATORILY emit the #1681 urgent candidate block
  (`urgency: main-red` + `failing_test: <node id>` + `wf_fix:`) whenever the gate excludes
  a file to get past a collection or red failure; a prose "noted for /daily follow-up" is
  not an acceptable terminal disposition. Mirror the same sentence into
  `.claude/rules/workflow-fix-on-bug.md` § Recursion guard "Urgent fast path".

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- `.claude/skills/issue/SKILL.md` (Step 10d merge-gate lost-update guard; Step 9c
  mandatory urgent-park emission)
- `.claude/rules/workflow-fix-on-bug.md` (§ Recursion guard "Urgent fast path" mirror)
- `tests/test_workflow_lint.py` (symbol + bundle-registration pin)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- The restoration must not revert #1698's own intended `workflow_lint.py` change — verify
  both #1701's and #1698's content are present on the resulting tree.
- `uv run pytest --collect-only -q` must exit rc=0 before this task's Step 9c gate runs;
  the gate itself cannot classify until it does.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 1007e9704d5f

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: B-P1, B-P2, J-P1.

---
title: Step 10d Guard 4 (lost-update refusal) is vacuous on the prescribed call path
  — --main-sha passes the main TIP, helper reads it as the merge-base
kind: infra
tags:
- wf-fix
created_at: '2026-08-20T19:19:38Z'
has_clean_result: false
parent_id: 2212
origin_prompt: 'Found while re-driving /issue 2212 Step 10d after a watcher completed-unmerged-respawn:
  Guard 4 returned GUARD4=pass on a branch whose scripts/workflow_lint.py snapshot
  dropped 8 main-added lines; the same helper refused when given the true merge-base.'
workflow: v1
---
# Guard 4 (lost-update refusal) is VACUOUS on the call path Step 10d prescribes — `--main-sha` is the main TIP, the helper reads it as the merge-base

## Goal

Fix the caller/helper contract mismatch that makes Step 10d's Guard 4 — the
mechanical backstop against a branch silently reverting already-merged sibling
work (#1701 -> #1698, encoded as a guard in #1713) — pass vacuously on every
branch, then add a mechanical pin so the vacuity cannot silently return.

## The bug

`scripts/step10d_guards.sh --guard 4` derives its comparison base as:

```
if [ -n "$MAIN_SHA" ]; then
    MB="$MAIN_SHA"
else
    MB=$(git -C "$WT" merge-base HEAD origin/main)
fi
```

i.e. it treats `--main-sha` as the **merge-base**, and its own `--help` text
documents the flag that way ("Guard 4 only: pinned merge-base").

But the Step 10d caller passes the `origin/main` **TIP**. In
`.claude/skills/issue/steps/18-step-10d.md`, the Guard-1 block sets

```
MAIN_SHA=$(git -C "$WT" rev-parse origin/main)
```

and the prescribed Guard-4 call is

```
GUARD4_OUT=$(bash scripts/step10d_guards.sh <N> --guard 4 --main-sha "$MAIN_SHA"); GUARD4_RC=$?
```

So the guard enumerates "lines `origin/main` ADDED since `<the origin/main
tip>`" — main diffed against itself — which is the EMPTY set. With no
main-added lines to look for, the "is each one present in the branch's copy?"
predicate is vacuously satisfied for every path on every branch, and the guard
emits `GUARD4=pass` unconditionally.

## Evidence (measured, not inferred)

Reproduced on task #2212's branch `issue-2212` at pre-merge tip `0eb8309f86`
(before its recovery merge), same helper, same tip, three invocations:

| Invocation | Verdict |
|---|---|
| `--guard 4 --main-sha <origin/main TIP>` (the prescribed form) | `GUARD4=pass` |
| `--guard 4 --main-sha <true merge-base>` | `GUARD4=refused`, `LOST_UPDATE_PATHS=scripts/workflow_lint.py(8)` |
| `--guard 4` (no flag; helper derives the merge-base) | `GUARD4=refused`, `LOST_UPDATE_PATHS=scripts/workflow_lint.py(8)` |

Hand-confirmed independently of the helper, on that same pre-merge tip:

- `scripts/workflow_lint.py` — `origin/main` added 17 lines since the
  merge-base; **8 were absent** from the branch's snapshot.
- `.claude/agent-memory/critic/MEMORY.md` — 139 main-added lines, **1 absent**.

So the true verdict was REFUSE and the prescribed call returned pass. #2212's
own prior Step 10d turn recorded `GUARD4=pass` / `LOST_UPDATE_PATHS=none` for
that branch — the same vacuous read, in a real merge attempt.

Blast radius at the time: a `--squash` landing of that branch would have
reverted 8 lines of a **bundled-lint** file (`workflow_lint.py`), plus stale
copies of 8 other zero-authorship sibling-synced files that `main` had advanced
2-9 commits past. That is precisely the #1701 -> #1698 class the guard exists to
refuse.

Not caught by anything else: Guard 3 explicitly EXCLUDES files whose only
branch-side touch is a `sync workflow-surface specs from` commit (correctly —
they are imported from main), so the stale-snapshot case is Guard 4's sole
responsibility.

## Acceptance criteria

1. The prescribed Step 10d Guard-4 call refuses on a branch carrying a stale
   whole-file snapshot of a workflow-surface file. Fix ONE side of the contract
   (either the helper computes `merge-base HEAD "$MAIN_SHA"` from a passed tip,
   or the Step 10d caller passes the merge-base) — and make the flag name /
   `--help` text agree with whichever semantics is chosen. Prefer the
   helper-side fix: the flag exists so the caller can PIN the main snapshot
   against fleet churn, which a tip is the right value for, and it keeps the
   no-flag path (already correct) and the pinned path in agreement.
2. A mechanical pin (a test in `tests/`) that FAILS if Guard 4 returns `pass`
   on a fixture branch whose snapshot drops a main-added line in the guarded
   scope — i.e. the vacuity itself is pinned, not just the current output. The
   fixture must exercise the flag form the Step 10d caller actually uses.
3. `.claude/skills/issue/steps/18-step-10d.md` Guard-4 block and the helper's
   usage/`--help` text state the same contract, so the next reader cannot
   re-introduce the mismatch.
4. Audit for the same tip-vs-merge-base confusion in the helper's OTHER
   `--main-sha` consumer (the `divergence` guard also takes/emits a
   `MAIN_SHA`) and state the finding either way.

## Notes for whoever picks this up

- Do NOT "fix" this by having the Step 10d caller drop `--main-sha`. That would
  work, but it discards the pinning the flag exists for (`origin/main` is a
  shared ref a concurrent session's fetch can advance mid-guard — the #1128
  rationale in the Guard-1 block).
- The two probes above are cheap to re-derive on any behind-main branch that
  has a spec-freshness sync commit touching `scripts/workflow_lint.py`.
- #2212 itself is NOT blocked on this: its landing was made safe by
  construction (every non-payload path reset to `main`'s exact blob, own diff
  reduced to the 5 approved deliverables, and Guard 4 then passes on the TRUE
  merge-base too). This task is the general fix.

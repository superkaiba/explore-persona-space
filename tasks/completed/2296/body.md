---
title: 'Step 10d lint gate''s mapped-invariant baseline leg runs pytest in the shared
  repo root — #2015 stash cycle kills it and false-NEWs pre-existing reds'
kind: infra
tags: []
created_at: '2026-08-14T15:45:59Z'
has_clean_result: false
parent_id: 2288
origin_prompt: 'measured on #2288 gate RUN1: baseline leg truncated at ~41% / 230
  bytes with 104 pre-commit patch files bracketing its window; gated leg clean; tg-new-nodes
  falsely listed the #2223 pre-existing thread-caps red'
workflow: v1
---
---
kind: infra
---

# Step 10d lint gate's mapped-invariant BASELINE leg runs pytest inside the shared repo root — the #2015 stash cycle kills it, and its death makes pre-existing reds classify NEW

## Goal

Move the Step 10d pre-push lint gate's mapped-invariant **baseline** pytest leg
off the shared repo root and onto a detached scratch worktree at the resolved
base. Today that leg runs `cd "$REPO_ROOT" && uv run pytest ...` in the tree that
~15 concurrent sessions mutate, so it is killed by ordinary fleet activity — and
when it dies the gate cannot classify pre-existing reds, so it fails closed with
`crash` (or, worse, marks a pre-existing red NEW by construction).

## Mechanism

`.claude/skills/issue/SKILL.md` Step 10d gate workload, mapped-invariant legs:

```bash
# BASELINE leg — root copy on the payload-free main tree
( cd "$REPO_ROOT" && timeout --kill-after=30s ${TG_T}s ... uv run pytest "${TG_BASE_TESTS[@]}" ... ) \
  > /tmp/issue-<N>-tg-baseline.txt 2>&1 || TG_BASE_RC=$?
```

Every fleet commit's pre-commit stash cycle runs `git checkout -- .` repo-wide
for the hook window (`.claude/rules/repo-root-uncommitted-state.md`, #2015:
`staged_files_only.py:23,81`). A pytest process whose source tree is reverted and
restored underneath it mid-collection/mid-run does not survive. The GATED leg is
unaffected — it runs `cd "$WT"` in the issue worktree, which no other session
touches.

Downstream consequence is worse than the crash itself: `tg-new-nodes.txt` is
computed as gated-hits minus baseline-hits, so an empty/aborted baseline makes
EVERY gated hit look NEW. A pre-existing red then reads as payload-attributed —
the same false-NEW shape as #2293, arrived at through a different surface.

## Measured incident (#2288, 2026-08-14)

Gate RUN1 returned `crash`. Forensics:

- baseline leg output truncated at ~41% progress, 230 bytes, NO summary line
  (bound was `TG_T=960s`; it died at ~4 min, so not a timeout)
- gated leg completed cleanly: `1 failed, 291 passed in 275.87s`, the single
  failure being the #2223 pre-existing thread-caps red
- `tg-new-nodes.txt` = `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
  — i.e. a KNOWN pre-existing red classified NEW, purely because the baseline
  had no result to subtract
- lint legs were clean and correctly attributed: `lint-owndiff.txt` and
  `lint-new.txt` both EMPTY (the #2223 lint red present in both baseline and
  gated logs, correctly subtracted)
- **104 pre-commit `patch<epoch>-<pid>` files bracket the baseline leg's window**
  (08:28:00–08:36:30, ≈13 stash cycles/min) — the #2015 diagnostic tell

So the gate's verdict logic behaved correctly (fail closed on an untrustworthy
compare); the DEFECT is that the baseline leg is structurally unable to complete
on a busy shared root.

## Proposed fix

Run the baseline leg in a detached scratch worktree checked out at the RESOLVED
BASE — the same instrument `step9c_baseline.py` already has
(`create_scratch_worktree`, sparse-cone profile, `gate_tmp_root()` placement) —
rather than in `$REPO_ROOT`. Two notes for the round:

1. Reuse the existing helper rather than minting a second scratch-tree
   implementation; #2293 is concurrently fixing that helper's base selection, so
   sequence behind it or coordinate the base argument.
2. The baseline leg's SEMANTICS are "the payload-free tree at the merge base" —
   a scratch worktree at the resolved base satisfies that strictly better than
   the root (whose local `main` can also be divergent — the #2293 finding).

## Acceptance

1. The Step 10d baseline leg no longer executes pytest with cwd inside the
   shared repo root.
2. With a fleet-representative commit rate against the root (or a simulated
   stash cycle), the baseline leg completes and the gate returns a
   `pass`/`block` verdict rather than `crash`.
3. A pre-existing red present in BOTH trees is subtracted (absent from
   `tg-new-nodes.txt`) rather than classified NEW.
4. The gated leg's cwd (`$WT`) is unchanged.
5. `tests/test_issue_skill_gate_recipe_hardening.py` (or the nearest prose-pin
   test for this recipe) pins the new cwd so the shared-root form cannot return.

## Provenance

Surfaced by #2288's Step 10d gate RUN1 (`epm:progress` markers of 2026-08-14).
Sibling: **#2293** — same false-NEW outcome, different surface
(`step9c_baseline.py compare`'s pristine oracle cut from root HEAD instead of the
resolved base). Both belong to one family: *the gate's baseline oracle must not be
taken from the contended, possibly-divergent shared repo root.* Consider whether
one round should fix both; they are filed separately because the target files
differ (`.claude/skills/issue/SKILL.md` here, `scripts/step9c_baseline.py` there)
and CLAUDE.md dedups workflow-fix candidates on `(target_file, fingerprint)`.

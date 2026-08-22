---
title: /issue Step 10d merge fence never binds $WT — git -C '' retargets every worktree
  op at the SHARED repo root (destroys a valid gate verdict; reaches a shared-root
  working-tree revert)
kind: infra
tags: []
created_at: '2026-08-14T23:41:14Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2293 Step 10d: the extracted safe-case merge fence uses
  git -C "$WT" throughout but only re-derives REPO_ROOT; git -C '''' is a documented
  no-op, so the verdict conjunct compared the certified sha against the repo root''s
  main HEAD and the BLOCKED arm rm -f''d a valid pass verdict.'
workflow: v1
---
# `/issue` Step 10d safe-case merge fence: `$WT` is never bound, and `git -C ""` silently retargets every worktree operation at the SHARED repo root

## Goal

Make the Step 10d safe-case merge block in `.claude/skills/issue/SKILL.md`
(the `#### The auto-merge procedure (safe case…)` fence, currently lines
~12925-13360) bind `$WT` itself, and fail fast if it is unset — closing a gap
that today (a) destroys a VALID lint-gate verdict and forces an extra ~84-minute
gate run, and (b) routes a working-tree revert at the SHARED repo root, below
the `guard_repo_root_branch.sh` hook's visibility.

## The gap

The fence re-derives `REPO_ROOT` inline and says why:

```bash
# REPO_ROOT is re-derived inline — fenced blocks are separate shells, and the
# guards block's derivation is not in scope here:
REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
```

So the separate-shell problem was known and fixed — for `REPO_ROOT`. `$WT` gets
no such treatment, yet the same fence uses it far more heavily: the Guard-0/1
push, the verdict conjunct, the whole post-gate Step 5a re-sync, the re-bind
probe, and the head-sync pre-check. The fence even WARNS about `$WT` binding
("Uses the ALREADY-BOUND `$WT` (do NOT re-derive from cwd; a repo-root cwd would
rebind to the shared root)") — correct advice with no binding behind it.

**`git -C ""` is a documented no-op**, not an error: git treats an empty `-C`
argument as "the current directory". So an unbound `$WT` does not crash — every
`git -C "$WT" …` in the fence silently retargets the orchestrator's cwd, which
at Step 10d is the SHARED REPO ROOT on `main`.

## Observed on #2293 (2026-08-14)

The fence was extracted mechanically from SKILL.md and run as a script — the
shape the recipe itself prescribes for the lint gate ("compose the script with
the Write tool… the launcher-only bg-Bash"). Two consequences fired:

**1. A valid gate verdict was destroyed (this happened).** The merge conditional's
third conjunct is

```bash
[ "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt)" = "$(git -C "$WT" rev-parse HEAD)" ]
```

With `$WT` empty this compared the certified sha `17c398f303` against the repo
ROOT's `main` HEAD `97fc0f8596` — unequal by construction, for every branch, on
every run. The BLOCKED arm then ran its `rm -f /tmp/issue-<N>-lint-verdict.txt`
and consumed a `pass` verdict that was correct: the printed diagnostic reads
`(verdict: pass\n17c398f303…)` while refusing the merge for a "stale sha". Cost:
one destroyed verdict plus a third ~84-minute gate run to regenerate bytes the
gate had already produced (the `#1082` never-hand-write rule correctly forbids
restoring them from the gate log).

Note the shape: the *diagnostic prints the evidence that contradicts it*. That is
what makes this expensive rather than merely wrong — it reads as a genuine
staleness and invites a re-run rather than a diagnosis.

**2. A shared-root working-tree revert is reachable (this did NOT fire, by luck).**
The post-gate re-sync runs

```bash
git -C "$WT" checkout origin/main -- $SAFE_SPECS_10D
git -C "$WT" commit -m "…spec-freshness…" -- $SAFE_SPECS_10D
```

With `$WT` empty that is a `git checkout <pathspec>` **at the shared repo root** —
a working-tree revert across `.claude/agents .claude/agent-memory .claude/skills
.claude/rules .claude/workflow.yaml CLAUDE.md scripts/workflow_lint.py …`, which
CLAUDE.md forbids outright and `guard_repo_root_branch.sh` exists to block. It
would discard concurrent sessions' uncommitted edits to those paths (the
2026-06-01 / #815 / #841 incident class), then COMMIT the result at the root.
**The PreToolUse hook cannot see it**: the hook matches Bash command TEXT, and
this runs inside a script file, so the guard is bypassed entirely. On #2293 the
block exited at the verdict conjunct before reaching the re-sync — fail-closed
ordering saved it, not design.

The Guard-0/1 push clause degrades quietly too: `git -C "" push origin issue-<N>`
pushes from the root. On #2293 it printed "Everything up-to-date" (the branch was
already pushed), but a local Guard-0 memory commit or Guard-1 strip commit in the
worktree would NOT have reached the PR head ref — reintroducing exactly the #787
failure the clause exists to prevent, silently.

## Proposed fix

1. **Bind `$WT` in the fence, next to the existing `REPO_ROOT` derivation**, using
   the same construction the guards prelude uses (NOT a cwd derivation, per the
   fence's own warning):

   ```bash
   REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
   WT="$REPO_ROOT/.claude/worktrees/issue-<N>"
   ```

   Equivalently `eval "$(bash scripts/step10d_guards.sh <N> --guard prelude)"`,
   which already emits both and is the canonical spelling (#1978).
2. **Fail fast on an unusable `$WT`** at the top of the fence, so the degradation
   can never be silent:

   ```bash
   [ -n "$WT" ] && [ -d "$WT" ] || { echo "FATAL: WT unbound or missing ($WT) — refusing; every git -C \"\" would retarget the SHARED repo root" >&2; exit 1; }
   ```
3. **Audit the other fences for the same shape.** Any fenced block that uses
   `git -C "$WT"` without binding `$WT` in-fence has this bug latently; the
   merge-conflict recovery and artifact-confirmed procedures are the first places
   to check.
4. **Consider a lint check**: a `workflow_lint.py` pass over `SKILL.md` fenced
   bash blocks flagging any block that USES `$WT` (or `$REPO_ROOT`) without
   binding it in the same fence. That generalizes past this one site and is
   mechanically checkable.

## Acceptance

1. The Step 10d safe-case merge fence binds `$WT` in-fence and refuses with a
   FATAL diagnostic when it is unbound or not a directory.
2. A pin test asserts the fence text contains both the binding and the guard
   (the `tests/test_issue_skill_*` family is the natural home).
3. Any other SKILL.md fence using `git -C "$WT"` without an in-fence binding is
   either fixed the same way or explicitly documented as prelude-dependent.
4. No change to the merge forms, guard semantics, verdict-consumption rules, or
   the `#1082` never-hand-write contract.

## Provenance

Surfaced by #2293's own Step 10d merge (task #2293 `epm:progress`, 2026-08-14).
#2293's subject matter is closely related but distinct: it fixes the Step 9c
pristine oracle being cut at `git_head(root)` instead of the resolved diff base,
in `scripts/step9c_baseline.py`. This task is the SKILL.md merge fence, a
different file and a different mechanism — though the SAME underlying error, the
shared repo root's HEAD silently substituted for the intended tree's. Distinct
fingerprint from #2303 (Step 5a sync data dependencies + unchecked commit rc) and
#2297 (Step 9c launcher argv-newline split).

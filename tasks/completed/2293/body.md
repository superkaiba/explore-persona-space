---
title: Step 9c pristine oracle cut from repo-root HEAD instead of the resolved diff
  base — root-divergence window yields false NEW classifications
kind: infra
tags: []
created_at: '2026-08-14T13:49:56Z'
has_clean_result: false
parent_id: 2288
origin_prompt: 'surfaced by #2288''s Step 9c compare: scratch_sha 058f7e8e10 (a task
  #2254 root-lineage commit) lacked scripts/issue2223_r5_pubtopic.py which IS on origin/main,
  so 5 pre-existing reds classified NEW; create_scratch_worktree uses sha = git_head(root),
  not the resolved base'
workflow: v1
---
---
kind: infra
---

# Step 9c pristine oracle is cut from the repo ROOT's local HEAD, not the resolved diff base — a root-divergence window yields false NEW classifications fleet-wide

## Goal

Make `scripts/step9c_baseline.py compare --run-pristine` cut its pristine oracle
at the RESOLVED DIFF BASE (fetched `origin/main` semantics, or an explicit
`--base` REF), not at the shared repo root's local `main` HEAD. Today, whenever
the root's local `main` transiently diverges from `origin/main` — routine on a
shared root that ~15 concurrent sessions commit to — the oracle is cut from a
lineage that can LACK commits already on `origin/main`, and every gate node
failing because of those commits is misclassified **NEW**. A false NEW is
fail-closed: it walls the gate and forces a manual provenance override.

## Mechanism (read from source, not inferred)

`scripts/step9c_baseline.py`:

- `create_scratch_worktree(root, wt_cones, timeout_s)` — docstring: *"Materialize
  a detached SPARSE scratch tree at **root's HEAD**"*; body: **`sha = git_head(root)`**,
  then `git worktree add --detach --no-checkout <tree> <sha>`.
- `_selector_context()` DOES resolve a correct base (`sel.resolve_base(DEFAULT_BASE
  ='origin/main', wt, fetch=False)`, honoring an explicit `--base`), but that base
  feeds only `compute_touched` / selection — it never reaches the oracle checkout.
- Consequence: **`--base` cannot correct the oracle.** Both oracle paths are
  root-lineage — the scratch path (`git_head(root)`) and the #1408 clean-root
  degradation path (the root working tree itself).

## Observed incident (#2288, 2026-08-14)

#2288's gate run measured `5 failed, 5847 passed, 12 skipped` (2722.64s). All 5
failures were caused by `scripts/issue2223_*.py` files **already on `origin/main`**
(its tip WAS the #2223 r5 commit `6d29131458`). The compare nonetheless returned
all 5 as `new`, because its scratch oracle was cut at `058f7e8e10` — a
`task #2254 epm:progress` commit on the root's then-divergent local lineage which
does NOT contain `scripts/issue2223_r5_pubtopic.py`:

```
scratch_sha: 058f7e8e10a404f396c14a3b0eeaac6883544815
git cat-file -e 058f7e8e10a4:scripts/issue2223_r5_pubtopic.py  -> ABSENT
git cat-file -e origin/main:scripts/issue2223_r5_pubtopic.py   -> PRESENT
git merge-base --is-ancestor 058f7e8e10a4 origin/main          -> NO
git merge-base issue-2288 origin/main                          -> 6d29131458 (the #2223 r5 commit)
git log --format=%h origin/main..issue-2288                    -> exactly 2 commits, both #2288's
```

So relative to the TRUE merge target the branch adds only its own 2 commits, yet
relative to the oracle's base it appeared to introduce #2223's entire r2/r3/r5
payload. The root self-reconciled ~1h later (`main...origin/main` = 0/0, clean),
and a re-run cut the oracle at a #2223-bearing HEAD — so the misclassification is
purely a function of WHEN the compare runs relative to the root's sync state.
Cost in that one instance: ~1h of pristine-run wall spent producing an
unusable verdict, plus a second ~1h re-run.

## Why this is fleet-wide, not a one-off

The window is structural, not exceptional: `task.py` commits to the shared root
and pushes on every marker post across ~15 sessions, and `sync_repo_root.py`
reconciliation is not instantaneous. ANY session whose branch was rebased onto an
`origin/main` newer than the root's current local `main` — the normal state for a
branch that rebased recently — gets every node touched by the intervening commits
classified NEW. The #2206 precedent already records the cost shape of a false NEW:
"~1h wall + manual provenance override".

## Proposed fix

Thread the resolved base into the oracle: `create_scratch_worktree` takes the
base SHA (resolved exactly as `_selector_context` resolves it, honoring `--base`)
instead of calling `git_head(root)`, and the JSON records BOTH `scratch_sha` and
the base ref it came from so a reader can audit the choice. The #1408 clean-root
degradation path needs the same treatment — a clean root is not a CORRECT root, so
"clean" must stop being sufficient grounds to use the root working tree as oracle
(gate it on `git_head(root) == resolved_base`, else force the scratch path).

Design question for the round to settle explicitly: whether an oracle base that is
not an ancestor of the worktree's HEAD should be `indeterminate` rather than
silently used — the fail-safe direction for a base the branch never descended from.

## Acceptance

1. With the root's local `main` deliberately behind `origin/main` (simulate: a
   detached root-lineage checkout, or a fixture repo), a node failing ONLY because
   of a commit present on `origin/main` and absent from the root's HEAD classifies
   **pre-existing**, not NEW.
2. `--base <REF>` demonstrably controls the oracle checkout; a regression test
   pins `scratch_sha` to the resolved base rather than `git_head(root)`.
3. The clean-root degradation path is gated on root-HEAD-equals-resolved-base.
4. The compare JSON records the oracle's base ref alongside `scratch_sha`.
5. No change to the existing strip / ledger / paired-oracle semantics.

## Provenance

Surfaced by #2288's Step 9c gate (task #2288, `epm:progress` markers of
2026-08-14). Sibling filing: #2289 covers the #2223 SCRIPTS' own two defects (a
bare `hf_hub_download(` reding the no-flags lint bundle, and module-top torch
imports before `load_dotenv()`); THIS task covers the ORACLE that misattributed
them. The two are independent: fixing #2289 removes these particular 5 reds,
fixing this task removes the false-NEW channel for every future divergence
window.

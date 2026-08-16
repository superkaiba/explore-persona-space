---
name: worktree-porcelain-newline-fail-closed
description: git worktree list --porcelain splits newline-bearing paths into orphan lines (NO C-quoting on git 2.34.1) — deletion-gate parsers must parse in RECORD form and return None on any unrecognized line; .strip() corrupts trailing-space paths
metadata:
  type: feedback
---

`git worktree list --porcelain` (git 2.34.1, verified by live repro in #2147)
emits worktree paths RAW — space, tab, backslash, double-quote all survive
unescaped on one line, and there is NO C-quoting (decoding C-quotes is dead
code on this git). But a path containing a NEWLINE necessarily SPLITS its
record: the `worktree ` line carries a TRUNCATED path and the remainder lands
as an orphan continuation line. A line-wise `startswith("worktree ")` parse
records the truncated path, so the REAL registered path is ABSENT from the
set — a positive non-registration proof built on that set is fail-OPEN and a
REGISTERED worktree can reach `shutil.rmtree` (#2147 r4, Codex R3-C1/SIB-1:
severity right, mechanism wrong — Codex attributed it to C-quoting).

**Why:** one truncated record in the listing poisons the WHOLE set; any
consumer that treats non-membership as license must refuse the entire parse.

**How to apply:** parse in RECORD form — records open with `worktree <path>`
(path VERBATIM after the prefix; never `.strip()`, which corrupts
trailing-space paths into a ghost entry) and close at a blank line; the only
recognized in-record lines are `HEAD `, `branch `/`detached` (one slot),
`bare`, `locked[ reason]`, `prunable[ reason]`, each at most once. ANY other
non-blank line, attribute-outside-record, or duplicated slot ⇒ return `None`
(whole listing ambiguous), with every caller treating `None` as
refuse-to-act. Slot rules alone are NOT enough (#2147 r5, coordinator repro):
a path embedding `\nbare` yields a continuation line that IS a valid absent
flag — `bare`+`detached` coexist because a genuine detached record simply
lacks `bare`. Close it with a positive EXISTENCE cross-check at record
close: every parsed path must exist on disk as a directory, missing-dir
tolerated ONLY when the record carries `prunable` (git 2.34.1 verified: a
deleted worktree dir lists `prunable gitdir file points to non-existent
location`). Canonical impl + 9-test battery:
`scripts/clean_experiment_downloads.py::_registered_worktree_paths` +
`tests/test_vm_disk_guard_slurm_src.py::test_r4_* / test_r5_*`. Known latent siblings
(read-only parsers, no deletion licensing, left as-is in #2147):
`scripts/verify_task_body.py::_parse_worktree_list`,
`scripts/audit_stranded_task_commits.py::list_worktrees` — harden them the
same way if they ever feed a destructive or licensing decision. Pre-fix
demonstration technique for importlib-by-path script tests: [[prefix-scratch-git-show-demo]].

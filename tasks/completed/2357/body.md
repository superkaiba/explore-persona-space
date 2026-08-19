---
title: 'guard_root_code_commit.sh: pathspec-limited root commits blocked by foreign
  staged code files (scoping defect)'
kind: infra
tags:
- trigger-dense
created_at: '2026-08-18T00:17:55Z'
has_clean_result: false
workflow: v1
---
# guard_root_code_commit.sh: pathspec-limited root commits are blocked by FOREIGN staged code files (scoping defect)

**Observed (2026-08-18, session driving /issue 2332):** three successively simpler commit shapes at the repo root were ALL blocked by `guard_root_code_commit.sh` with `BLOCKED: repo-root commit carries UNCERTIFIED code payload: tests/test_issue2094_rev_butler.py` — a file staged by ANOTHER session (cert-diag: `binding=staged want=6b02dd57c79c staged=6b02dd57c79c worktree=6b02dd57c79c cert=stale:267588s>max_age:21600s`), never named in any of my pathspecs:

1. `cd <root> && cp ... && git add <memory-file> && git commit -m "..." -- <memory-file>` (compound)
2. separate call: `cd <root> && git commit -m "..." -- <memory-file>` (staging done in a prior call)
3. the guard's own prescribed remediation shape verbatim — bare pathspec-limited commit, unquoted path, plain redirections.

The guard's remediation text states: "a pathspec-limited commit is never blocked by foreign staged files ... the guard scopes its check to the pathspec." Shape (3) contradicts that promise — the scoping fails, plausibly because the command carries a `cd <root> && ` prefix (the shell cwd resets between calls, so a cd prefix is the NORMAL committer shape fleet-wide), or because the pathspec parser mis-handles the `-- <path>` extraction in the presence of output redirections placed after the pathspec.

**Impact:** while ANY session leaves a stale-certified code file staged at the shared root (a standing condition — cert max age is 6h and staging can sit much longer), EVERY other session's root commits are blocked regardless of pathspec, forcing `EPM_ALLOW_ROOT_CODE_COMMIT=1` overrides (I took one for a non-code `.claude/agent-memory/**` commit; reason recorded here). Overrides normalize exactly the bypass the guard exists to prevent.

**Fix direction:** make the pathspec-scoping arm robust to (a) `cd X && git commit` compound prefixes, (b) redirections after the pathspec, (c) verify with a fixture that a foreign staged uncertified code file + a clean pathspec-limited commit of a non-code path passes. Also consider: a stale cert on a CONTENT-MATCHING staged file (want==staged==worktree) arguably warrants a re-certify hint rather than a hard block attribution to unrelated committers.

**Repro state:** foreign file `tests/test_issue2094_rev_butler.py` staged at root with cert-stale as of 2026-08-18 ~00:5xZ; blocked commands and full guard outputs in session `cmswf8cr4iox3wo0upn2gxj6q` (issue-2332 worktree) transcript.

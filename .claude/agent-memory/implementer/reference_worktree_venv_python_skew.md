---
name: worktree-venv-python-version-skew
description: Worktree .venv can build at a different python (3.12) than root's (3.11) — exception-message test expectations flip verdicts between the two trees; attribute newly-red full-suite entries both-ways in the SAME venv
metadata:
  type: reference
---

A worktree's freshly-built `.venv` can resolve a DIFFERENT python minor
version than the repo root's (`root .venv = python3.11`, `issue-2214
worktree .venv = python3.12`, observed 2026-08-20). Tests pinning
exception MESSAGES then flip verdicts between the two trees: #2329's
`pytest.raises(Exception, match=r"[Pp]ickle")` around `torch.save(lambda)`
passed at root (py3.11 `PicklingError: Can't pickle ...`) and failed in the
worktree (py3.12 `AttributeError: Can't get local object ...<lambda>` — no
"pickle" substring).

**Why:** uv picks the interpreter per-project at venv creation; a worktree
venv built later lands on a newer system python.

**How to apply:** when diffing a worktree full-suite failure set against a
repo-root baseline (the A6 shape), a "newly-red" entry must be attributed
BOTH-WAYS in the SAME tree + venv (swap only the diffed file via
`git show origin/main:<path>`), never by comparing across the two venvs —
a root-green/worktree-red split can be pure interpreter skew. Related:
[[worktree-pytest-resolves-main-src]].

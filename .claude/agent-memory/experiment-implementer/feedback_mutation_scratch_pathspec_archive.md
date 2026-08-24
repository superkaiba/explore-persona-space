---
name: mutation-scratch-pathspec-archive
description: Mutation-verification scratch trees — pathspec-limit `git archive` (full tree is multi-GB w/ eval_results); scoped list_repo_tree for the ~1M-file data repo
metadata:
  type: feedback
---

Two wedges hit in one round (#2329 r17), both with one-line fixes:

1. **`git archive <sha> | tar -x` of the FULL tree times out** in this repo —
   the tree object includes all of `eval_results/` (multi-GB) even from a
   sparse worktree. Pathspec-limit it to what tests need:
   `git -C "$WT" archive <sha> scripts tests src pyproject.toml | tar -x -C <scratch>`
   (tests/conftest.py rides the `tests` pathspec; run pytest with the
   worktree's `.venv/bin/python` and cwd=scratch — the editable
   `explore_persona_space` install resolves to the worktree src, fine when
   mutations live in `scripts/`).
2. **`snapshot_download(allow_patterns=...)` on the data repo hangs** — it
   lists the WHOLE ~1M-file repo before filtering (hung past 480 s). Use
   scoped `HfApi().list_repo_tree(..., path_in_repo=<prefix>)` + per-file
   `hf_hub_download` (48 files in seconds).

**Why:** mutation proofs are required per-item on #2329-class rounds
("neuter the guard in a scratch git archive tree, never the worktree") and
schema-from-artifact probes need real staged files; both wedges burn a
whole Bash timeout each if re-derived.

**How to apply:** any mutation-verification round (scratch archive) or any
probe/staging pull against `superkaiba1/explore-persona-space-data`.

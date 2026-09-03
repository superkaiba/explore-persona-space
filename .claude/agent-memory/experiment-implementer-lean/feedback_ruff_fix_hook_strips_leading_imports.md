---
name: ruff-fix-hook-strips-leading-imports
description: A PostToolUse ruff-fix hook deletes imports added BEFORE their usages exist — add the code that uses them first, then the imports; guard-red proofs can run in-process via pytest.MonkeyPatch, no file swap
metadata:
  type: feedback
---

Two mechanics from #2658 group-E round 3 (2026-09-02):

1. **Edit ordering under the format hook.** A PostToolUse hook runs ruff
   autofix after every Edit on this repo's test/script files; an `import re`
   / `from x import Y` added while still unused is silently DELETED by the
   very next hook pass (my import edit was reverted before the test body
   landed). Order edits: add the code that references the symbol FIRST, then
   the import. **Why:** ruff F401 autofix removes unused imports; the hook
   sees each Edit in isolation. **How to apply:** any multi-edit round adding
   both imports and their consumers to one file.

2. **Prove a guard test fires red WITHOUT touching the worktree.** Load the
   test module by path (`importlib.util.spec_from_file_location`), build
   `pytest.MonkeyPatch()` manually, and perturb in-process: monkeypatch the
   payload builder to drop a tracked key, or inject a synthetic signature
   parameter via `fn.__signature__ = sig.replace(parameters=[...+Parameter(
   "synthetic_new_bar", KEYWORD_ONLY, default=...)])`. Zero file mutation, so
   no restore/`git diff --stat` verification burden and no race with live
   sibling rounds — strictly better than the swap-blob-and-restore probe when
   the redness you need is behavioral, not missing-symbol. (Missing-symbol
   fails-pre-fix still needs the pinned /tmp tree of BASE_SHA blobs.)

Related: [[judge-pilot-report-resume-fields]], [[shared-worktree-partial-stage-commit]].

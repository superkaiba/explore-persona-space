---
name: referenced-helper-not-in-head-tree
description: Eval/training scripts that `importlib.util.spec_from_file_location` an external helper module by absolute path die with FileNotFoundError at relaunch when the helper was added to local working tree but never `git add`'d. Distinguishable from missing-data because traceback frame is `_spec.loader.exec_module(_mod)` / `get_data` in `<frozen importlib._bootstrap_external>`. Verify with `git ls-tree -r HEAD --name-only | grep <basename>` — if empty, code-class. Burned at #408 eval-only v11 relaunch (HEAD 3a152cbb5) when `_generate_corpus_length_distribution_figure` referenced `scripts/issue_377_plot_corpus_lengths.py` that wasn't committed.
metadata:
  type: feedback
---

When a relaunch script crashes seconds after `[phase=eval]` with:

```
FileNotFoundError: [Errno 2] No such file or directory: '<repo>/scripts/<helper>.py'
  File "<frozen importlib._bootstrap_external>", line 1130, in get_data
```

the immediate question is: **is the helper missing from HEAD or just unavailable on this pod?** Three checks, in order:

1. `git ls-tree -r HEAD --name-only | grep <basename>` — empty → never committed → **code-class**.
2. `find <repo> -maxdepth 4 -name '<basename>'` — present on disk but not in tree → stale local edit shipped via worktree but not the issue branch.
3. `git log --oneline -5 -- <path>` — empty → confirms (1).

**Why this matters at launch:** the broken `importlib.util.spec_from_file_location(...).loader.exec_module(...)` pattern bypasses normal Python `import` machinery. It does NOT show up in `ruff check`, `pytest --collect-only`, or any pre-flight that exercises module imports — `spec_from_file_location` is called at runtime only when the dispatcher's code path hits it. So the implementer can write+test the helper locally, commit only the caller (eval_issue408.py) by accident, push the branch, and code-reviewer PASS the diff. The crash only surfaces on a fresh pod that pulled the branch.

**Reflexive bounce as code-class** (not infra): the launcher, pod state, GPUs, disk, env are all healthy. The fault is missing source. Failure note should call out the exact tracable line + the empty `git ls-tree` grep result so the next implementer round doesn't waste a turn re-launching.

**Preferred remediation (for implementer):** rather than "commit the missing helper", prefer guarding the call: wrap the `spec_from_file_location` in a try/FileNotFoundError that logs+skips when the helper is absent. Decorative figures (corpus-length-distribution, etc.) should never fail-close on a long eval run that's about to consume an H100 for hours. If the helper is load-bearing (an actual eval step, not a diagnostic plot), then commit it AND add a `tests/test_issue<N>_helper_paths.py` that imports it the normal Python way so future renames/moves break tests at code-review time, not at relaunch.

**Pre-launch grep to add at experimenter prelaunch** (cheap): `grep -nE 'spec_from_file_location|exec_module\\(' scripts/<dispatcher>.py | head -5` — if non-zero matches, ssh `ls -la <each_referenced_path>` before the `setsid nohup`. Don't waste a 60s subagent turn launching a process that dies on import.

Related: [[wrapper-pipefail]] (different class, but same pattern of "the crash is real and the wrapper masks it"); [[brief-phase-all-mismatch]] (same family of "v11 just landed and something downstream broke"); [[stale-eval-proc-steals-log]] (also a pre-launch checklist item).

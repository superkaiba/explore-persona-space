---
name: referenced-helper-not-in-head-tree
description: Scripts that spec_from_file_location an external helper die with FileNotFoundError from <frozen importlib._bootstrap_external> at relaunch when the helper was never committed; ruff/pytest/code-review can't see it. Verify via git ls-tree.
metadata:
  type: feedback
---

A relaunch crashing seconds into a phase with `FileNotFoundError: ... scripts/<helper>.py` from `<frozen importlib._bootstrap_external>` means the dispatcher loads a helper via `importlib.util.spec_from_file_location(...).exec_module(...)` — a pattern that bypasses normal import machinery, so ruff, `pytest --collect-only`, and code review all miss a never-committed file. The implementer tests locally, commits only the caller, and the crash surfaces only on a fresh pod.

**Why:** #408 eval-only v11 relaunch (HEAD 3a152cbb5) — `_generate_corpus_length_distribution_figure` referenced `scripts/issue_377_plot_corpus_lengths.py`, absent from the tree.

**How to apply:**
1. Triage in order: `git ls-tree -r HEAD --name-only | grep <basename>` (empty → never committed → **code-class**); `find <repo> -maxdepth 4 -name <basename>` (on disk but not in tree → stale local edit); `git log --oneline -5 -- <path>`.
2. Cheap pre-launch grep: `grep -nE 'spec_from_file_location|exec_module\(' scripts/<dispatcher>.py` — on matches, `ls` each referenced path on the pod before the nohup.
3. In the failure note, recommend: decorative helpers (diagnostic plots) get a try/FileNotFoundError log-and-skip so they never fail-close an hours-long eval; load-bearing helpers get committed PLUS a test importing them the normal way so future moves break at review time.

---
name: untracked-twin-add-certification
description: Certify a "committed as clean add of an untracked file" claim by diffing the committed blob against the live untracked twin; attribute extra hunks in a mechanical sweep by ruff-format-probing the PARENT blob
metadata:
  type: feedback
---

Two probes that settled #2183 R1 g1 (19-file mechanical PROJECT_ROOT sweep with 2 adds):

1. **Untracked-twin diff.** When a commit claims a new file is "the untracked
   repo-root file, committed as a clean add with only the fix applied", do not
   read the 700-line add — run
   `diff <(git show <sha>:scripts/<f>.py) <repo-root>/scripts/<f>.py` against
   the live untracked twin. Output confined to the fix region == the whole
   body is certified byte-identical in one probe. Pair with
   `git ls-tree origin/main -- <f>` (empty) to certify the "absent from main"
   half of the claim.
2. **Extra-hunk attribution via parent-blob format probe.** A lone
   off-template hunk in a mechanical sweep (e.g. an f-string quote flip) is
   either scope creep or a formatter artifact. Extract the PARENT blob
   (`git show <sha>^:<f> > /tmp/x.py`) and run
   `ruff format --check --config pyproject.toml /tmp/x.py` — "Would reformat"
   proves the implementer's format pass forced the hunk; then it is a benign
   note, not a finding.

**Why:** both replace an unreadable-scale read (whole add / whole file
history) with a one-command certification; the twin diff also catches a
smuggled body edit that a header-only review would miss.

**How to apply:** any sweep commit whose message records an
untracked-file-add deviation, or any pure-mechanical commit with one hunk
that does not match the template. Related: [[mechanical_sweep_commit_review_recipe]].

Also from this round: ruff run on blobs extracted to /tmp uses DEFAULT config
(no pyproject upward walk) — 110 phantom errors; re-run from the worktree or
pass `--config` before reading any ruff output as a finding.

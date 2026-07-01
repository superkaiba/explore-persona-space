---
name: A follow-up round may RESUME partial prior work — diff worktree HEAD before reimplementing
description: The worktree branch can already carry a prior implementer's committed round + stranded uncommitted files; git-log/diff HEAD first, then ADD the missing piece, never re-build committed code
type: feedback
---

A same-issue follow-up round (revision_round=1) is NOT always a clean start. The
`issue-<N>` worktree branch can already carry a PRIOR implementer round's commit
(the earlier session did work, committed, then died / truncated) PLUS its stranded
UNCOMMITTED helper files.

**Why:** implementer sessions die mid-round (API refusal, truncation, watcher
respawn). The orchestrator re-dispatches "round 1" but the branch already has a
`task #<N> followup: ...` commit + untracked `scripts/issue<N>_*.py` from the
dead session. Building the whole plan from scratch then DUPLICATES committed code
and, worse, a parallel reimplementation DIVERGES from the reviewed committed
version (different signatures, different conventions) — clobbering it is a
regression.

**How to apply (do this BEFORE writing any code on a follow-up round):**
- `git -C <WT> log --oneline -8` — look for a `task #<N> followup:` commit already
  on the branch.
- `git -C <WT> status --short` — untracked `scripts/issue<N>_*.py` are a prior
  round's stranded work (read them; they're often integral — a dispatch calls them).
- For each file the plan names, `git show HEAD:<path>` — if the committed version
  already implements it, do NOT reimplement. ADD only the genuinely-missing piece
  (e.g. #778 r1: Leg A + null-battery seam were committed; only Leg B + the
  dispatch + a no-adapter sentinel + the reused-artifact staging were missing).
- Align new code to the COMMITTED conventions (loader signatures, out-tag scheme,
  the path a sibling upload script globs), not to your own fresh design.
- BUT verify the committed code is actually CORRECT against the plan — a prior
  round can commit a stale/wrong version. #778 r1: the committed null-battery still
  used the OLD 48-test BH pooling; the plan's reconciled fix was a 24-test
  stochastic-only split. That was a real defect to fix, not duplicate work. So:
  reuse committed structure, but re-read it against the plan's §-numbered
  requirements and fix genuine divergences (surgical edit, keep the committed shape).
- Commit the prior round's stranded-but-integral untracked files together with
  your additions so the deliverable is coherent (name the provenance in the report
  `(d)` so the reviewer knows some files came from a dead prior session).

Sibling: `feedback_reused_script_may_have_uncommitted_sibling_edits.md` (a reused
helper's working-tree copy can have a parallel session's UNCOMMITTED signature
change). Both are "diff before you trust the tree" lessons.

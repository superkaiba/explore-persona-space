---
name: Claude interp-critic raises verify_task_body.py FAIL as its sole REVISE — out-of-scope + often stale
description: When Claude's ONLY interp-critic REVISE driver is a verify_task_body.py mechanical-conformance FAIL (check 21 Body-Parameters-subset-of-methodology-doc-§2), re-run the verifier on the CURRENT body+doc yourself and check scope — it's the clean-result-critic's gate, not the interpretation lens, and the FAIL is frequently stale.
type: feedback
---

When Claude's `interpretation-critic` returns REVISE but its rationale concedes
the interpretation content is PASS-quality ("the interpretation content itself
is sound", "binding r1 fix landed correctly") and the SOLE blocking item is a
`verify_task_body.py` mechanical-conformance FAIL — almost always check 21
(`Body Parameters ⊆ methodology doc §2`) — side with the Codex PASS unless you
independently reproduce the FAIL on the CURRENT artifact.

**Why:** Two independent reasons converge on PASS, both verified at #559 r2
(2026-06-19):
1. **Scope.** The interpretation-critic's lens is interpretation honesty /
   plot-prose match / raw-sample plausibility. A `verify_task_body.py`
   Parameters-table-conformance FAIL is the clean-result-critic's gate (Step
   9a-bis) + the orchestrator's standing "FAIL blocks" rule — NOT an analyzer
   interpretation revision. The fix is a methodology-writer re-extend + body
   re-pin, NOT an interpretation edit. Siding with the REVISE spawns an
   analyzer round 3 that is a no-op on interpretation content and a write on
   methodology-writer's surface — a misroute. (Claude itself usually flags
   "owned by methodology-writer, not an interpretation edit" while STILL
   driving REVISE off it — that admission is the tell.)
2. **Staleness / non-reproduction.** The cited FAIL frequently does NOT
   reproduce on the live artifact. At #559 the methodology doc was already
   re-extended (commit 47b43faecb, §2.4 added) ~30 min BEFORE Claude's
   critique; Claude read a stale/un-refreshed worktree copy and reported a
   FAIL the live verifier never produces. And for ANALYSIS-ONLY tasks
   (kind: experiment but no model training) check 21 is *skipped* entirely —
   there is no training-hyperparameter table for the body Parameters to be a
   subset of; the body Parameters ("Task type", "Cross-behavior predictor",
   "Judges", "Trained-side DV", "Analysis") are analysis-design descriptors,
   and the table records `Training hyperparameters | n/a`.

**How to apply:**
- ALWAYS run the verifier yourself before believing the FAIL:
  `uv run python scripts/verify_task_body.py --issue <N> --methodology-doc <worktree-doc-path>`
  and read the OVERALL line. If it PASSes (or the cited check is skipped),
  Claude's blocker is discarded as "does not reproduce" / mistaken.
- Check whether the methodology doc was re-extended (a `methodology: EXTEND #<N>`
  commit) AFTER the round's body Parameters were rewritten — `git log --oneline
  -- docs/methodology/issue_<N>.md` + compare timestamps to the critique marker.
- A check 21 skip on an analysis-only task is correct behavior, not a bug to fix.
- Verdict: PASS the interpretation-critic gate; let Step 9a-bis own the
  mechanical conformance check. Add a standing rec that clean-result-critic
  re-run the verifier so the stale-doc read can't recur downstream.
- Index note: the same check is labeled "check 21" in the verifier's current
  output and "check 32" in older marker quotes (index drift) — same check.

**The cross-tree stale-doc hazard is verifier-wide, not interp-only.** The
SAME #559 trap bit the CLEAN-RESULT-critic reconcile too (epm:review-reconcile
v5): a naive `verify_task_body.py --issue 559` from the repo root reports a
spurious check-21 FAIL because the repo root holds the OLD 25 KB methodology
doc while the EXTENDED 46 KB doc (commit 47b43faecb) lives un-merged on the
`issue-559` worktree branch. ALWAYS pass `--methodology-doc
<worktree>/docs/methodology/issue_<N>.md` before trusting OR distrusting a
check-21 result — in EITHER direction (a repo-root run can spuriously FAIL a
body that PASSes, OR spuriously PASS a stale body). At #559 the worktree run
correctly SKIPPED check 21 (analysis-only) → OVERALL PASS.

Datapoints:
- #559 r2 interp-critic — Claude REVISE (verify_body FAIL sole driver, stale
  doc) vs Codex PASS → reconciled PASS (epm:review-reconcile v4).
- #559 r2 clean-result-critic — Claude PASS vs Codex needs_targeted_fix
  (data-access-blocked, sandbox DNS couldn't reach HF; tagged `[lens]` =
  mechanizable-verifier-blocked, NOT a content finding) → reconciled PASS
  (epm:review-reconcile v5). Reconciler ran `list_repo_files` from the VM:
  both pinned revs resolve + carry cited paths; procedurally-blocked-verifier
  strip rule → orchestrator-verifies-inline, no REVISE round. Lesson:
  a Codex `data-access-blocked` / DNS-failure flag is an infra strip, not a
  blocker, whenever the orchestrator/VM has the access Codex's sandbox lacked
  and the body content is unchanged.

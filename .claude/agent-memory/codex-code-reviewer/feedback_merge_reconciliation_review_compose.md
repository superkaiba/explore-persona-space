---
name: merge-reconciliation-review-compose
description: Composing a Step 10d divergence-reconciliation merge review (#2253 r3) — parent-relative scoped diffs, zero-hand-edit premise as check 1, misleading-range warning, adapted gate-scope note
metadata:
  type: feedback
---

For a Step 10d pre-merge divergence-delta reconciliation round (#1771→#2201), the review is a MERGE-SEMANTICS check, not a feature review. Compose pattern (validated #2253 r3, 2026-08-21):

- **Lead with the misleading-range warning.** `git diff <branch-tip>..HEAD` after a merge of main is main's own history (380 commits / ~29.8 MB on #2253) — ban the unscoped read explicitly, citing #521 (Codex flagged main-drift as churn and burned a reconciler round).
- **Zero-hand-edit premise is check 1, verified not assumed.** `git diff-tree --cc HEAD` empty + `log -1 --format='%H %P'` parent pins. Every downstream "nothing to read" claim rests on it; if it fails, everything escalates.
- **Diff acquisition = 4 scoped parent-relative diffs** (merge-base..each-parent for contributions; each-parent..HEAD for what the merge carries), all restricted to the gate-flagged overlap files. Dropped/mutated hunk in either direction = Critical `substantive`. Header literal: `sha-range <merge-sha> (merge, parent-relative, N-file scope)`.
- **Set-equality registrations get an own-enumeration duty** (tuple vs manifest read statically), never "the pin test passed" — the asymmetric-drop hazard is exactly what auto-merges hide.
- **Gate-scope on a pure merge:** the report legitimately carries an adapted `Gate-scope note` (no hand-written lines ⇒ no changed literals to sweep). Score PRESENT-but-adapted → at most CONCERNS; the diff-consistency half collapses into the zero-hand-edit check.
- Round-matched `epm:results` marker DID exist for the reconciliation round (posted by the gate's dispatch) — probe events.jsonl before assuming the follow-up-round placeholder path.

**Why:** the gate's own rationale is that a semantic collision can merge textually clean; the composed prompt must make Codex answer that question and nothing else, or it drowns in main's history.

**How to apply:** any round whose brief names a reconciliation/merge commit and parent SHAs. See also [[revision-round-compose-recipe]] and [[stale-base-mb-pin-and-fixture-remeasure]].

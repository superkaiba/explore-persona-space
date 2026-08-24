---
name: compose-time-ledger-snapshot
description: Inline the compose-time open-concern ledger snapshot into the prompt; Codex cannot run task.py — and readlink plan.md before labeling the inlined plan version
metadata:
  type: feedback
---

Two compose-time wins from #2254 r3 (2026-08-24):

1. **Inline the open-concern ledger snapshot (ids + severities + one-line
   summaries) into the prompt as compose-time ground truth.** Codex's sandbox
   cannot run `task.py list-concerns` (branch-guard + uv unavailable), so a
   "verify the body's open-concern acknowledgments" check is only scoreable
   if the composer hands over the snapshot. Doing so surfaced a real
   discrepancy pre-dispatch: the body claimed "nine remain open" and cited
   `firstk-empty-regen-cap-policy-bypass` as open while the ledger showed 8
   open and that id closed — turned into a concrete round-specific check with
   quoted-line requirements.
   **Why:** an unverifiable ledger claim otherwise degrades to prose-trusting.
   **How to apply:** whenever the interpretation/body acknowledges concern ids,
   run `list-concerns --open-only --json` at compose time and inline the id
   set + count into a round-specific check.

2. **`readlink plans/plan.md` before labeling the inlined plan version.** An
   `ls | tail -5` of the plans dir showed v5–v9, but the symlink resolved to
   v10.md — the envelope header would have mislabeled the plan version.
   **How to apply:** never infer the plan version from a truncated listing;
   readlink the symlink and state that version in the envelope header.

Related: [[cross-worktree-path-split]] (this round: worktree plan stale at v7,
worktree body.md stale too — passed /tmp copy of canonical body; all 7 figure
PNGs + sidecars + 3 reads JSONs blob-MATCHED the body pin, so worktree paths
were passable for those).

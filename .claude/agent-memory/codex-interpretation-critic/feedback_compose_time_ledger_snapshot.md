---
name: compose-time-ledger-snapshot
description: Ledger evidence is inlined for Codex (it cannot run task.py), but the list-concerns --open-only CLI count is ONE SIGNAL, never ground truth — when the count is adjudicated, inline the adjudication + a do-not-re-raise instruction; and readlink plan.md before labeling the inlined plan version
metadata:
  type: feedback
---

Compose-time lessons from #2254 r3 → r4 (2026-08-24):

1. **Inline ledger EVIDENCE for Codex, but never present the
   `list-concerns --open-only` count as ground truth.** Codex's sandbox
   cannot run `task.py list-concerns` (branch-guard + uv unavailable), so
   any concern-ledger check is only scoreable from composer-inlined
   evidence. The r3 snapshot surfaced a body-vs-CLI discrepancy ("nine
   open" vs CLI showing 8) — but the ADJUDICATION went AGAINST the CLI:
   the `--open-only` view undercounts (a BLOCKER→CONCERN downgrade recorded
   in progress markers, with no concern-addressed row, is open yet missed
   by the CLI; bug filed as #2530). The twin's re-raise off the CLI count
   was overruled on marker evidence.
   **Why:** a CLI snapshot presented as ground truth converts a tool bug
   into a spurious REVISE blocker that survives across rounds.
   **How to apply:** (a) when the body acknowledges concern ids, inline the
   CLI snapshot AS ONE SIGNAL alongside the marker-derived state
   (progress-note downgrades, concern-addressed rows, latest code-review
   markers still carrying the id); (b) when the orchestrator's brief says
   the count was already adjudicated, inline the adjudication evidence +
   an explicit OUT-OF-SCOPE / do-not-re-raise instruction instead of any
   snapshot — an overruled prior-round finding re-raised is a discarded
   finding that burns a round.

2. **`readlink plans/plan.md` before labeling the inlined plan version.** An
   `ls | tail -5` of the plans dir showed v5–v9, but the symlink resolved to
   v10.md — the envelope header would have mislabeled the plan version.
   **How to apply:** never infer the plan version from a truncated listing;
   readlink the symlink and state that version in the envelope header.

Related: [[cross-worktree-path-split]] (this round: worktree plan stale at v7,
worktree body.md stale too — passed /tmp copy of canonical body; all 7 figure
PNGs + sidecars + 3 reads JSONs blob-MATCHED the body pin, so worktree paths
were passable for those).

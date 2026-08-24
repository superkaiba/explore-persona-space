---
name: concern-row-ledger-grammar
description: Always instruct Codex to emit CONCERN:: rows in the strict ledger grammar (SEVERITY id summary) — pipe-delimited / key=value rows are rejected by the ledger validator
metadata:
  type: feedback
---

Every composed interp-critique prompt must instruct the exact
machine-readable concern-row grammar:
`CONCERN:: <BLOCKER|CONCERN|NIT> <lowercase-kebab-id like codex-interp-r<k>-c<n>> <one-line plain-prose summary>`
— severity uppercase from the three-value enum, id sequential from c1, no
pipes, no `id=` / `lens=` / `blocking=` key-value tokens, one row per
finding, all rows inside the marker block, no rows on a clean PASS.

**Why:** #2479 round 1 (2026-08-24) — the twin emitted
`id=.. | lens=.. | blocking=..` pipe-grammar rows; the ledger validator
(`persist_verdict_concerns.py` path) rejected them and the orchestrator had
to hand-normalize all 11 before they could persist.

**How to apply:** add a STRICT grammar block right after the output-format
template in every composed prompt (round 1 included — don't wait for a
rejection), with a literal example row and the explicit anti-patterns
(pipes, key=value fields) named. See [[cross-worktree-path-split-for-figures-vs-eval-jsons]]
for the sibling path-composition duties on the same rounds.

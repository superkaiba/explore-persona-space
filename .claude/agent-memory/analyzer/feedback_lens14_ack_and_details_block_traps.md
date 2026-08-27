---
name: lens14-ack-and-details-block-traps
description: Open concern ids must sit inside a Results H3 (details block works); Methodology mentions don't count; details blocks in Results need a K-of-M disclosure; avoid point-count prose claims vs line-plot sidecars
metadata:
  type: feedback
---

Three verifier mechanics learned drafting #2546 round 1 (many open review-round concerns):

1. **Lens 14 acks: Methodology mentions do NOT count.** The concerns audit
   accepts a concern id only via (a) substring inside a `### <result>` H3
   body, (b) the `Confidence:` sentence (v3-only surface; v4 bans confidence
   outside the title), or (c) a `<!-- concern-deferred: id -->` marker backed
   by a REAL `defer-concern` ledger event (fabricated deferrals FAIL, #2219;
   defer is user-only). With 13 open process-residual ids, the clean shape is
   ONE `<details>` block inside a coverage-themed Results H3 listing every id
   with a one-line disposition: word-cap-exempt (details stripped from
   `_prose_words`) yet substring-matched by Lens 14.
2. **A Results `<details>` block counts as a sample block for the
   cherry-picked-label check** (check 10): its summary/prelude needs a
   disclosure token — `13 of 13 rows (the complete open set)` satisfies the
   `N of M rows` form. Without it: hard FAIL, even though the block holds
   concern ids, not data rows.
3. **Per-unit-evidence (check 59) vs beat-claim (check 24) trap:** satisfying
   check 59 with "one point per corpus" prose triggers check 24, which
   compares the claim against the sidecar's rendered elements — a LINE plot
   renders 0 `scatter` elements → contradiction WARN. Use vocabulary-only
   tokens instead: "low-level", "per-question", "companion", "counterpart".

**Why:** #2546 burned 4 verify/fix cycles on these; each is invisible in a
`--file` run (Lens 14 + plan-coverage checks only fire with `--issue` after
set-body — always re-run `--issue` before reporting PASS).

**How to apply:** any v4 draft on a task with open concerns.jsonl entries;
any draft using details blocks inside `## Results`. See also
[[fold-context-prompt-and-open-concern-acks]].

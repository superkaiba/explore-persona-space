---
name: Plan-derived floors vs realized artifact grain
description: Verify the GRAIN (row/line counts) of row-grain-consuming reused artifacts before encoding plan floors as hard asserts — existence checks don't count rows; hard-kill only at the mathematical minimum, WARN+flag above it
type: feedback
---

#1900 fellows job 16055 died on `assert len(rows) >= 40` (anchor-mix floor):
the plan assumed "~50–300 positive rows/mix" but the reused
`delta_tf/<mix>/pos.jsonl` files carry EXACTLY 20 rows per content mix at the
pin — the fact-checker had verified the files EXIST, never their row counts.

**Why:** a reuse row's fitness includes its realized grain, not just its
resolution; a floor written against an assumed range converts a data property
into a crash.

**How to apply:** when implementing a plan whose asserts encode row-count
floors over reused inputs, PROBE the realized counts first (a one-file
`hf_hub_download` + line count is seconds); encode hard-kill only at the
mathematical minimum the computation needs (e.g. even/odd halves ⇒ ≥4/side),
and make plan-level quality floors a LOUD WARN + persisted flag (the #1900
fix: `check_anchor_mix_floor`, commit `c6737b1e`). Sibling probe duty: check
EVERY mix family the code reads (the marker mix had a different filename +
1000→200 filtered rows).

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Cross-frame gate asserts vs manifest selection reads](feedback_cross_frame_gate_asserts.md) — gates frame-matched or frame-free (direction+floor); record the other frame, never equality-assert it (#1900 r6)

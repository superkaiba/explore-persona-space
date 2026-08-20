---
name: fork-hardcoded-parent-constants-probe
description: Model-swap fork review — certify every hardcoded parent-table constant (realized m, unit counts, sub-floor cells) by probing the COMMITTED parent artifact with the code's exact filter semantics; re-derive computed ceilings live (#2389 R1 g4)
metadata:
  type: feedback
---

A model-swap fork (`issue<child>_analysis.py` forked from `issue<parent>_analysis.py`)
carries hardcoded parent-run facts: `PARENT_REALIZED_M`, `PARENT_P1_UNIT_COUNT`,
docstring claims like "3 of the 16 sit below the floor: cellX=1, cellY=7, cellZ=8".
Each is a checkable claim against a committed artifact — never take the fork's
comment or the plan's restatement as evidence.

**Why:** #2389 R1 g4: three probes settled three claim families in one call each —
(1) `jq '.family_m' eval_results/issue_2329/f_metrics/stats.json` certified
`PARENT_REALIZED_M={28,12,27}`; (2) a 10-line python over the committed 2162
`f_cells.jsonl` certified the 31-unit count-assert AND the per-cell sub-floor
docstring — crucially using the code's OWN kept-filter (`arm=="steered"`,
`f_beh is not None`, `|separation|>=0.5`), since a naive row count gives different
numbers; (3) re-running the fork's `_derive_ce_ceilings()` logic live gave
{P1:16, P2:8, P3:14} — matching the plan's P1=16 and exposing the module
docstring's "16/7/14" off-by-one guess (computed value was right; prose wrong).

**How to apply:** for each hardcoded parent constant, find the committed artifact
it summarizes and recompute it THERE with the fork's exact selection semantics
(copy the filter lines, don't paraphrase). For any "derived, never hardcoded"
constant, run the derivation standalone (import only the bank/registry module,
not the fork — the fork's run-module import chain may be dirty in the worktree).
Docstring numeric guesses beside derived constants are the cheap catch.
Sibling of [[stats-reuse-driver-live-probes]] (live probes settle claims) and
[[perfile-id-namespace-not-leakage]] (recompute-verify split artifacts).

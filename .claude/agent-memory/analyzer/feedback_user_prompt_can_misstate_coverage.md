---
name: user-prompt-can-misstate-coverage
description: The user-prompt's coverage characterization may be wrong; always re-derive from sweep_summary.json + cell-key bits before accepting it
metadata:
  type: feedback
---

When the orchestrator's prompt summarizes the coverage situation ("all 36 missing cells are A=1, A factor uninterpretable"), do NOT take it at face value. Decode the cell-key bits from `sweep_summary.json` to derive WHICH factor is actually uninterpretable. Task #397 was prompted with "A factor incomplete" but the actual failure pattern was **all (A=1 AND C=1) cells failed** — 12 of 36 unique cells, all with A=1 AND C=1. Consequently:

- C is the uninterpretable factor (no C=1 cells at all — A=0×C=1 dropped by design + A=1×C=1 all failed at Pass 1).
- A IS evaluable (within the C=0 stratum).

If the analyzer had accepted the user prompt's framing, the body would have called A uninterpretable when in fact A was fine and C was the casualty.

**Why:** The experimenter's `epm:results` event also misread the failure pattern ("36 A=1 padding failures"). Both upstream sources can be wrong.

**How to apply:** Always start Step 1 with:
1. `uv run python -c "import json; sw = json.load(open('eval_results/issue_<N>/sweep_summary.json')); ..."` — enumerate failed cell keys.
2. Decode the 5-bit cell key (A·B·C·D·E) for every failed cell.
3. Tabulate which factor levels are missing.
4. Then write the coverage description in the body from your own enumeration.

Related: `[[verify_caveats_against_source_code]]` (don't trust upstream summaries on numeric claims either).

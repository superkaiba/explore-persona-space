---
name: verbatim-embed reject taxonomy before budget fix
description: "Before fixing a verbatim-embed yield floor as 'token-cap truncation', measure the reject taxonomy on the persisted raw rows (finish_reason + answer-length quantiles) — #1345 cps: only 13/1,778 zero-rejects were cap-truncated; the dominant mode was model-side abandonment of LONG answers at temp 1.0"
type: feedback
---

When a verbatim-embedding gen phase misses its yield floor and the triage says
"stories truncated at max_new_tokens", VERIFY on the persisted raw bundle
before sizing the fix: count `finish_reason == "length"` and compare
answer-token quantiles of rejects vs keeps. #1345 cps r4: the triage sampled
"49% contain the answer's 120-char prefix → truncated at 1024"; measured, only
13/1,778 zero-rejects hit the cap — 1,023 contained the prefix but ended with
`finish_reason=stop` (321 story-ends mid-answer, 309 paragraph-break cuts, 87
early quote-closes, 306 divergence; median reproduced fraction 24%), and
reject answers were 4x longer than keeps (median 337 vs 88 tokens).

**Why:** the model never sees max_tokens — a budget raise only fixes rows the
CAP cut (finish_reason=length). Long-verbatim-copy failure at temp 1.0 is the
model's own EOS/abridge choice (cumulative EOS hazard over a long quote +
length prior); the levers are temperature/greedy for copy fidelity or an
answer-length feasibility cap (recipe/planner decisions, not instrument fixes).

**How to apply:** any yield-floor fix round on a generation phase — the
finish_reason + length-quantile digest over the persisted raw rows takes
minutes off-pod (counts only, content-hygiene safe) and decides
budget-vs-recipe before code is written. Also: normalization tolerance (NFKC +
curly quotes + ws collapse, raw-offset-mapped, gate == extractor matcher)
recovered ~10% of zero-rejects — worth taking, but never the headline lever.
Worked impl: `c.find_verbatim_occurrences` + measurement transcript in #1345
`epm:experiment-implementation v13`.

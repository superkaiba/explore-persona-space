---
name: seed-replication-fold-cjk-recount
description: Fold rounds adding NEW judged pools (seed replications) owe their own Step 3.7 CJK scan; per_item_scores + generations.jsonl join makes the recount cheap and local.
metadata:
  type: feedback
---

A fold round that adds NEWLY JUDGED pools (a seed-N replication of deciding
cells, a re-judged arm) owes its OWN language-intrusion audit next to the new
adjudication — the parent body's audit covers only the parent pools.

**Why:** #2224 fold: the seed-137 pools carried 561/9,000 (6.2%) CJK
generations; the exclusion recount kept every substantive contrast direction
but flipped one near-zero contrast (+0.02 → −0.09), which the body must label
convention-dependent rather than silently count in the 16/18 agreement.

**How to apply:** the join is usually cheap and local — per-cell
`trait_scores.json` carries `trait_expression.per_item_scores` keyed
`<qid>-g<draw>` (null = unscored, skip), and the pod-harvested
`data/issue_<N>/**/postft_eval/<cell>/generations.jsonl` carries
`{qid, draw, response}`; flag CJK per (qid, draw), recount cell means +
paired contrast deltas over shared non-intruded keys (pure counting, no text
into context). Commit the audit JSON beside the round's eval artifacts and
quote `intruded/total` + the recount verdict in the seed-replication `###`
prose. See `scripts/issue2224_fu2_cjk_audit.py` ([[clean-result-critic-round-1-pre-flight-checklist]]).

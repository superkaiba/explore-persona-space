---
title: shared_fold_map.json on main is a 1,761-conv SMOKE map, not the 26,889-conv
  production map — guard the load
kind: infra
tags: []
created_at: '2026-08-12T16:42:26Z'
has_clean_result: false
parent_id: 2054
origin_prompt: run the gaps
workflow: v1
---
# The canonical `shared_fold_map.json` on main is a SMOKE artifact (1,761 convs), not the production map (26,889)

## Goal

Make `eval_results/issue_2054/shared_fold_map.json` on `main` either BE the production
fold map, or fail loud when loaded — so no future round silently fits on a 1,761-conversation
smoke map while believing it used the 26,889-conversation production one.

## The trap

`eval_results/issue_2054/shared_fold_map.json` on `main` reads:

```
n_conv_ids = 1761   fold_of = 1761 entries   k = 5   seed = 137
```

The production map #2054 actually ran on has `n_conv_ids = 26889` across 5 variants. Confirmed
independently from the banked per-cell fit JSONs on the HF data repo
(`issue2054_lattice/fits/*.json`), every one of which records
`fold_map: {k: 5, n_conv_ids: 26889, seed: 137}`.

So the artifact committed at the canonical path on `main` is a smoke map covering only
`char_helios`, while #2054's own promoted Methodology describes "5 conversation-grouped folds
from one shared fold map (26,889 conversations, seed 137) reused by every cell, so held-out
sets are identical across cells".

## Why it is dangerous rather than cosmetic

The seed (137) and k (5) MATCH the production map, so a consumer that sanity-checks those two
fields passes. The only field that reveals the substitution is the conversation count. A round
that loads this path gets:

- a fold map covering 473 of the 11,901 rows of the assistant cells (measured on the
  attrib_quoted inserted base cell), and
- silently reduced held-out sets that are NOT comparable to any banked #2054 number, despite
  the whole point of a shared fold map being cross-cell comparability.

This was hit live on 2026-08-12: the Gap A context-to-context round
(`scripts/issue2054_ctx2ctx_fit.py`) would have fit on the smoke map had its implementer not
noticed the count and routed to the production blob on the `origin/issue-2054` branch instead,
the way `issue2054_cross_render_fit._load_production_fold_map` does.

## Why the existing workaround is not sufficient

The parent's own loader reads the production map from the `origin/issue-2054` BRANCH BLOB, not
from `main`. That works for code that knows to call it, and every #2054 production number is
therefore sound — this task does NOT impugn any banked result. But it means the canonical
on-main path is a live trap for:

- any new round that reaches for the obvious path (the Gap A and Gap B rounds both did),
- any reader reconciling the promoted Methodology against the committed artifact, and
- anything that runs after `issue-2054` is eventually deleted or rewritten.

## Acceptance criteria

1. Decide the intended contract for the on-main path: either (a) commit the PRODUCTION map
   (26,889) at `eval_results/issue_2054/shared_fold_map.json`, or (b) rename the smoke artifact
   to something unmistakable (e.g. `shared_fold_map.SMOKE.json`) and leave the canonical name
   absent.
2. Whichever is chosen, add a loader-side guard so the failure is LOUD, not silent: a consumer
   loading a fold map must assert the conversation count against an expected floor (the
   `_load_production_fold_map` smoke-refusal floors already implement exactly this — promote
   them to the shared path rather than re-inventing).
3. Grep `scripts/` and `src/` for every reader of `shared_fold_map.json` and route each to the
   guarded loader; disposition any that legitimately want the smoke map.
4. Do NOT regenerate or alter the PRODUCTION map, and do NOT touch any banked per-cell fit JSON
   — every #2054 number was computed on the correct map and stays valid.

## Provenance

Found 2026-08-12 while running the user-requested zero-GPU follow-up rounds on #2054
(`epm:progress` v259/v261/v264 on that task). Not a #2054 result defect — an artifact/provenance
defect on `main` that threatens future rounds.

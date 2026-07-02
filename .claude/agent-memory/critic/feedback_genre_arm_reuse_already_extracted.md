---
name: Genre-arm c_C reuse — parent already extracted the matched pool
description: When a follow-up "recomputes fresh because the parent store is X-pinned", check whether the parent's OWN prior follow-up already ran the SAME matched pool and uploaded reusable tensors
type: feedback
---

When a genre/probe-pool amendment plan asserts "the parent store is
{Betley/X}-pinned so I must recompute c_C fresh," do NOT take the
parent's *currently-wired* loader as proof the matched pool was never
extracted. Read the parent issue's events: the parent often ran its OWN
same-issue follow-up over the EXACT matched pool already.

**Why:** #658 v2 (genre-generalization-ultrachat) recomputes c_C fresh
because `issue658_common.load_cc_last_store` is hard-pinned to the Betley
hash (`I594_PROBE_POOL_HASH = ad687bec…`). True — but #594 ALSO ran a
`probe-genre-generalization` follow-up (2026-06-11) over the SAME
`data/issue594/probes_ultrachat.json` pool and uploaded 52 tensors to
`superkaiba1/explore-persona-space-data/issue594_context_geometry/analysis_tensors_probegen`
including `context_vectors_mean.pt` (prompt-mean AND last-input-token c_C
on the UltraChat pool) + `per_probe/<id>.pt` for all 50 contexts. So the
UltraChat c_C the plan recomputes (~0.5 GPU-h) already exists on HF.

**How to apply:** This is a Concern, NOT a REVISE — recomputing to the
same recipe yields the identical tensor, so the conclusion does not
change; it is ~0.5 GPU-h of the ~14, well within The Bar's "cheaper is
not a REVISE." BUT flag it two ways: (1) reuse drops the cost; (2) better
— the reusable #594 UltraChat c_C is a FREE empirical stress-test of the
fresh-recompute (the agreement between the two validates the c_C
recipe-recompute, addressing the "is c_C genre-recompute faithful?"
question). The genuinely-new GPU work is answer-side v0(C) (G1/G2, ~3h)
+ the E0 battery (G6, ~4h) — #594 only captured PROMPT-side context
vectors, never the model's generated answer-side activations, so the
bulk of the run is correctly new.

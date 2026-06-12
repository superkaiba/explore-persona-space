---
name: Flag-A/B with reused control arm — approvable shape (#613)
description: Single-variable collator-flag A/B vs a reused parent arm — what made #613 v1 APPROVE-able; the stale code-comment trap on slot identity
type: feedback
---

Rule: a retrain-one-arm / reuse-the-other A/B is approvable when it carries: (1) the full a-g reuse fitness check with an explicit apply-and-read parity assert (re-read the reused arm's terminal, bounded vs committed numbers — #613 used 0.5 nat); (2) a frozen comparison band derived from the PARENT's seed gap, never recomputed with the new arm; (3) a step-1 manipulation check on the newly-live channel with smoke-named telemetry (WandB series + JSON channel presence); (4) registered branches for co-land / suppression / amplification AND "live but toothless" so every outcome is reportable.

**Why:** #613 (flag-on vs flag-off negative-loss placement, parent #601) shipped all four and APPROVE was correct — every confound left (cross-time generation drift on committed on-policy numbers, EOS-boosting shifting own-generation endpoints) was weighable from the four-float + R4 diagnostics.

**How to apply:** Check the code comment vs the plan's slot arithmetic — the #474 collator comment claims the negative loss slot is the "SAME slot the DV reads", but row construction (`R+"\n\n"+" ※"` positives vs bare `R` negatives) puts them one separator apart. A plan that trusts the comment instead of measuring both slots (sep-plain + sep-marker reads) loses the mechanism interpretation on a null. Also: registry seeds tuples can be stale (dense_200p800n carries `seeds=(137,)`; seed-42 came from a `--seed` override) — locate reused arms by explicit HF/JSON paths, never the registry tuple.

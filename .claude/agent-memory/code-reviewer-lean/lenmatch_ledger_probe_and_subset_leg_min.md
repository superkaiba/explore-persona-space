---
name: lenmatch-ledger-probe-and-subset-leg-min
description: Certify a matching/subsample analysis by pulling ONLY the small rows.json ledgers from HF and recomputing the selection layer; and check a pre-stated cross-leg min formula against the legs the code actually iterates (#2378 9a-ter)
metadata:
  type: feedback
---

Two lessons from the #2378 length-matched refit review (9a-ter inline round).

1. **Ledger-only HF pull certifies the whole selection layer.** When the staged
tensor store is already reaped, do not skip recomputation: the per-part
`rows.json` ledgers are tiny JSON — `hf_hub_download` just those (65 files),
symlink into a store-shaped dir, and run the driver's OWN functions
(`answer_token_lengths` → `matched_selection` → `control_selection`) against
the committed fold map. One probe simultaneously certified: matching table
exact-match, identical per-cell histograms, all 80 folds' n_train/n_eval,
per-leg medians, fold inheritance, AND that the fence-trip resume drifted
nothing (deterministic seeded selections reproduce the resumed cells'
artifacts). The fit layer then only needs the estimator-transcription diff vs
the named reviewed core. **Why:** the selection layer is where matching bugs
live, and it needs no tensors — the /tmp-rerun zero-diff pattern
([[tmp-rerun-zero-diff-analysis-artifact]]) at ~1 MB download instead of 2 GB.
**How to apply:** any subsample/matching/fold-subset round whose store is
gone — check whether the ledger alone determines the contested quantities.

2. **A pre-stated global-min formula implemented over a subset of legs.**
`k = min(1024, floor(0.5 * min n_train over all cells/LEGS/folds))` was
computed over MATCHED selections only, with a docstring claiming leg-invariance
("both legs share n"). False in general: same n, different fold composition ⇒
different per-fold minima (realized: matched 760 vs control 772 — harmless only
because matched happened to bind). Same family as
[[registered-gate-quantity-substituted]]: diff the computed quantity's
ITERATION DOMAIN against the formula's stated domain, then check the realized
minima of the OMITTED legs from the artifacts before crediting the formula.

---
name: cluster-null-port-blocks-cluster-intersect-cell
description: Porting the #2163 decoder-cosine cluster null to a new statistic - blocks = cluster INTERSECT conditioning cell; percolation census reported, not tuned (#1482)
metadata:
  type: feedback
---

Porting the #2163 cluster-preserving permutation null onto a statistic with
its OWN conditioning (crossed-strata cells, matched pairs): define permutation
blocks as cluster INTERSECT cell and feed the #2163 `_ConcatPerms` /
`_SizeExchangePerms` constructors per-FEATURE cell ids with `n_strata` = cell
count. The singleton reduction to the feature-grain within-cell shuffle then
holds exactly, and no cluster ever crosses a cell (report the split-cluster
count; independent sub-block movement loses cross-cell coupling, so the band
is a lower bound in that respect). Worked port:
`scripts/issue1482_cluster_null_concordance.py` (all gates 0-diff at n=120,716).

**Why:** #1482 banded the paper's stepwise concordance at each property's
selection round; the conditioning cells are categorical crossings, so #2163's
cluster-grain decile strata have no direct analog - the intersection-block
form is the faithful generalisation that keeps the machinery-validation gate
(singleton reduction) exact.

**How to apply:** also (1) permute y within cells via a rank identity for
binary/AUC-style statistics - rank(y_perm) = rank_pre[perm] since permutations
never cross cells, which batches the whole draw axis as one gather; (2) expect
PERCOLATION at low cosine thresholds on full-universe graphs (at tau=0.30 the
120,716-feature layer-19 BatchTopK decoder graph has a 71,484-feature giant
component, 59%; 33% at 0.35, 12% at 0.40) - report the census per tau and let
the band carry it, never tune tau; (3) size_exchange FREEZES any block unique
in (cell, size) - the giant component is always frozen, so its frozen-fraction
line (59% at 0.30) is the conservatism disclosure.

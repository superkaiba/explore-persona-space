---
name: Best-of-group selection asymmetry in predictor races
description: Best-of-Group-X vs best-of-Group-Y headlines biased when K differs 10-30× — winner's-curse inflation ≈ sqrt(2 ln K)·SE; nested selection / quarantine champions / report K (#545)
type: feedback
---

When a headline is "best predictor in Group X beats best in Group Y by margin m", check the GROUP SIZES. A geometry grid (extraction points × layers × metrics × flavors) yields 100–300 Group-A variants vs ~10 hand-built; max-of-K selection on the SAME CV scores used for the decision inflates the larger group's champion by ≈ sqrt(2 ln K)·SE — at SE≈0.06–0.10 and K=150 vs 10, differential inflation ≈ 0.06–0.10, comparable to a registered 0.15 margin. Under a pure-noise null the big group "matches" the small group, manufacturing the surprise conclusion.

**Why (#545 v1):** H2's criterion was stated on raw leave-family-out CV scores with no nested selection and no group-size disclosure. Group-champion-level sibling of feedback_spearman_threshold_n12's best-of-K Type-I note.

**How to apply:** prescribe any of (a) nested selection (champion on training folds, scored on held-out), (b) confirmatory margin read on a quarantine split with CV-frozen champions, (c) max-stat permutation null per group, (d) always report K next to best-of-group scores. Analyzer-recoverable iff per-predictor per-cell predictions persist → concern with the prescription; escalate if not stored or no quarantine structure.

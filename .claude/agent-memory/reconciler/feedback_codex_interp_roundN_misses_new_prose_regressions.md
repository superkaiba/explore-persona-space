---
name: Codex interp round-N misses regressions in NEW revision prose
description: On round-2+ interp splits, Codex anchors on round-(N-1) request closure and its "Regressions: None found" can miss a new false superlative added by the revision — recompute new closing/transition sentences yourself.
type: feedback
---

On round-2+ interpretation-critic adjudications, do not credit Codex's
"Regressions Introduced by the Revision: None found" — recompute any NEW
superlative/comparative sentence ("weakest throughout", "never", "always",
"only") the revision added, against the raw JSON, at the READ the adjacent
figure presents (best-layer vs layer-averaged is the recurring split).

**Why:** #812 r2 — Codex accurately verified all 8 round-1 requests closed
but PASSed a revision that ADDED "The PCA-reduced mean is weakest
throughout" to Result 2 (best-layer bars section). At the best-layer read
mean_pca was weakest for only 3/7 behaviors and BEAT the plain mean for
refusal (0.656 vs 0.612); the claim held only layer-averaged (7/7). Claude
caught it; reconciler recomputation matched Claude exactly → REVISE. Same
claim-vs-data class as that issue's round-1 "never exceeds".

**How to apply:** when Claude REVISEs on a prose-vs-data mismatch in NEW
round-N prose and Codex PASSes citing round-closure verification, weight
Codex's PASS as covering the OLD findings only; the disputed new sentence
must be verified from the JSON directly. Check which aggregation (per-layer
max vs per-layer mean) the sentence's home section/figure actually shows.

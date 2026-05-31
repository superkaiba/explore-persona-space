---
name: spearman-use-scipy-with-tie-correction
description: Hand-rolled rank-Pearson without tie correction gives wrong Spearman ρ when there are many ties in either axis; always use scipy.stats.spearmanr
metadata:
  type: feedback
---

When computing Spearman ρ, use `scipy.stats.spearmanr` — never a hand-rolled rank-Pearson, especially when ties are common.

Hand-rolled rank-Pearson assigns ranks 1..N to N items with no tie correction. When one axis is heavily saturated (e.g. 54 of 72 cells at substring rate = 1.00), the tied items all get distinct ranks under naive sort-based ranking, which is wrong. `scipy.stats.spearmanr` applies the standard mid-rank tie correction (Eq. 13 in Hollander/Wolfe), which can shift ρ substantially.

**Concrete example (task #397 round 1):** Hand-rolled rank-Pearson gave ρ = 0.26; `scipy.stats.spearmanr` gave ρ = 0.48 on the same data. Both interpretation critics independently caught the discrepancy and called it a numerical bug.

**Why:** With ties the average-rank convention is `(low_rank + high_rank) / 2` for each tied item; the bare sort-based ranking gives distinct ranks `low_rank, low_rank+1, ..., high_rank`, which biases the rank-correlation toward whatever stable-sort order happened to apply.

**How to apply:**
```python
from scipy import stats
import numpy as np
rho, p = stats.spearmanr(np.array(xs), np.array(ys))
```

Never:
```python
# WRONG — no tie correction
rx = sorted(range(n), key=lambda i: xs[i])
rank_x = [0]*n
for r, i in enumerate(rx): rank_x[i] = r+1
# ... rank-pearson on rank_x, rank_y
```

The hand-rolled version is only correct when both axes have all-distinct values. In ANY substring-rate / fuzzy-rate context with ceiling/floor saturation, ties are guaranteed and the correction matters.

Related: `[[verify_caveats_against_source_code]]` (a numerical claim that an independent critic can't reproduce is a bug, not a disagreement).

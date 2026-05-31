---
name: check-for-within-cell-clusters-before-reporting-mean
description: A "partial saturation" mean can hide multimodal cluster structure; always sort by selectivity and eyeball the distribution before reporting an average as the headline
metadata:
  type: feedback
---

When an aggregate "mean selectivity = 0.17" lands in the middle of a 0-to-1 range, do not report it as the headline number without first sorting the underlying cells by selectivity and checking the distribution shape.

**Concrete example (task #397 round 1):** I reported E1 (tail-32 loss) as "partial: source 0.83, bystander 0.66, selectivity 0.17". Independent critic (Claude) sorted the 24 E1 cells by selectivity and found a trimodal split: 3 cells at sel ~0.78 (clean), 16 at sel < 0.30 (lockstep failure), 3 at sel ~0 (dead). The 0.17 mean is the average across three qualitatively-distinct behaviour modes; it obscures the real structure and misleads the mentor reading the TL;DR.

**The fix:** before promoting a per-stratum mean to the TL;DR, run:

```python
# Sort per-cell selectivity within the stratum
sorted_cells = sorted(stratum_cells, key=lambda r: r['sel'], reverse=True)
# Eyeball the distribution
for r in sorted_cells:
    print(f"{r['cell']:>6} {r['src']:>10} sel={r['sel']:>7.3f}")
# Count cells in each behaviour-mode bin
n_clean = sum(1 for r in sorted_cells if r['sel'] > 0.5)
n_lockstep = sum(1 for r in sorted_cells if r['sel'] < 0.3 and r['src_rate'] > 0.5)
n_dead = sum(1 for r in sorted_cells if r['src_rate'] < 0.1 and r['bys_rate'] < 0.1)
```

If you see 3+ cells in any non-headline cluster, mention the cluster structure explicitly in the body and report cluster-conditional means alongside the overall mean.

**Why:** the headline-mean framing is what reviewers and the mentor actually read. A 0.17 mean across (0.78 × 3, 0.15 × 16, 0.00 × 3) tells a different story from "the regime is partially selective" — the real story is "the regime collapses for 16/24, works cleanly for 3/24, fails entirely for 3/24, and the 3 clean cells share substrate B=1 + D=1 which is a positive lead." Suppressing the structure costs the next experiment a free lead.

Related: `[[position_distribution_when_marker_eval]]` (same shape: aggregate rates hide qualitative structure that requires per-row inspection).

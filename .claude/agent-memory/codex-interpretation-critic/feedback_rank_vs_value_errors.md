---
name: Lens-7 rank vs value error pattern
description: Interpretation bodies sometimes state correct raw values but wrong rank claims in sample blocks — a class of error that only surfaces by independently recomputing ranks from the raw JSON
type: feedback
---

When a body provides sample blocks like "(pair X): value = 0.0205 (rank 16/171, small)", the raw VALUE may be correct while the RANK is wrong. This happened in issue #269: the 1-cosine values for (poet,comedian), (poet,villain), (villain,comedian) were exactly right (0.0205, 0.0240, 0.0183) but the claimed cosine ranks (16, 32, 9 out of 171) were systematically wrong — actual ranks were 103, 118, 81. The JS ranks were similarly wrong (165/171, 164/171, 158/171 claimed vs actual 140, 139, 137).

**Why:** The analyzer likely computed the ranks from a differently-ordered list or made a transcription error. Values come from direct matrix lookups but ranks require sorting the full upper triangle.

**How to apply:** For every sample block that claims "(value) (rank N/M)", independently recompute the rank by sorting the full upper-triangular list from the raw JSON and finding where the pair falls. Never trust stated ranks without verification — values are easier to copy correctly than rank positions.

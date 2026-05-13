---
name: Caption n= ambiguity: phrase count vs completion count
description: Figure captions that say "n trials labeled below cohort name" but the figure shows phrase counts, causing wrong total-completions claim
type: feedback
---

When a figure shows cohort bars labeled with n=40, n=60 etc., and the caption says "n trials labeled below cohort name; total X completions," verify whether the n= refers to phrases or completions. If n= is phrase count and each phrase has K completions, the caption total should be sum(n_phrases_per_cohort * K). Seen in issue #331 Figure 1: caption claimed 19,200 completions but actual was 18,400 (n= were phrase counts × 80 completions each, one cohort miscounted).

**Why:** Easy to miscalculate when summing "n phrases × 80 completions" especially when one cohort (famous, n=10) has a different cardinality.

**How to apply:** For every figure caption with "total N completions," independently compute sum(cohort_phrases * completions_per_phrase) from the JSON and verify it matches the caption claim.

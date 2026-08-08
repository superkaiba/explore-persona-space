---
name: stratified-null-not-centered-at-chance
description: A covariate-stratified permutation null for an AUROC/rate statistic is NOT centered at chance — test against the null's own mean, never against 0.5
metadata:
  type: feedback
---

When a permutation null is built by shuffling labels WITHIN strata of a
covariate (the activity-decile-stratified nulls this project uses everywhere),
the stratification deliberately PRESERVES the covariate-label association, so
the null statistic is **not centered at its chance value**. Test the observed
value against the null distribution's OWN centre:

```python
null_mid = float(np.mean(null_draws))
p = ((np.abs(null_draws - null_mid) >= abs(obs - null_mid)).sum() + 1) / (len(null_draws) + 1)
```

**Why:** testing `|obs - 0.5| >= |null - 0.5|` conflates two different
questions and silently inverts verdicts. #1482 full-width: `interpretable`
had AUROC 0.4885 against a stratified null centred at **0.4236** — the
observed sits far ABOVE its null band (enriched among the best-predicted
beyond activity), but the deviation-from-0.5 test returned p = 1.0000
("not significant") purely because the null itself is far below chance.
5 of 10 labels were mis-called this way; the corrected test flipped them to
p = 0.0005. `speaker_language` flipped the other way in meaning: AUROC 0.576
looks like a strong language effect but its null is 0.587, so it is
*below* what activity alone predicts.

**How to apply:** any AUROC / prevalence / rate statistic tested against a
stratified shuffle. ALWAYS persist the null mean and the above/below
direction next to the p-value — a reader cannot interpret the p without
knowing where the null sits. The studentized scan-corrected band (z against
per-k null mean/sd) is already correct by construction; it is the *global*
p that invites the 0.5 mistake. Related: [[selection-symmetric-nulls]],
[[rank-null-equals-observed-anchor-dominated]].

---
name: Scan all glob layouts for per-sample eval JSONs
description: Multi-seed eval directories often store per-seed detailed JSONs in inconsistent layouts (flat top-level, nested seed<N>/, nested seed<N>/eval_seed<N>/). Always scan ALL plausible patterns before reporting "n of N seeds available" to avoid silent data subsetting.
type: feedback
---

When parsing per-sample detailed JSONs across a multi-seed condition directory, scan ALL of these glob patterns and union the results before computing aggregate statistics:
- `<base>/seed<N>_<eval>_detailed.json` (flat at top level)
- `<base>/seed<N>/<eval>_detailed.json` (one level nested)
- `<base>/seed<N>/eval_seed<N>/<eval>_detailed.json` (two levels nested)
- `<base>/eval_seed<N>/<eval>_detailed.json` (eval-only nested)

**Why:** In the Aim 5.11 25% matrix evil_correct condition, only 5 of 10 seeds were caught by a `seed<N>/eval_seed<N>/...` glob because the other 5 had been written at the top level as `seed<N>_<eval>_detailed.json`. The reported coherence-filter row was a 5-seed cherry-pick that happened to be the higher-alignment half (28.95 vs 27.34). The pass-2 reviewer caught this; the corrected n=10 numbers (unfilt 28.15, filt 29.75, misrate 64.6%) shifted the misrate range from 61-70% to 64.6-70.2%. Qualitatively the null held; quantitatively a number that went into a published table was wrong.

**How to apply:** Whenever computing aggregate stats from per-seed detailed JSONs in any `<cond>_multiseed/` directory: (1) list ALL files under the directory tree first (`find <dir> -name "*detailed*"`), (2) verify the count matches `n_seeds`, (3) only then compute means/CIs. If the count does not match, investigate before reporting -- never silently report n < n_seeds without explicitly flagging which seeds are missing AND verifying they are missing from disk (not just missing from your glob).

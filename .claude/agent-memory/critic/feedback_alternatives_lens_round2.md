---
name: Alternatives lens — round 2 follow-through patterns
description: Three confound classes that survive a good v1→v2 alternatives revision — BPE-token boundary, negative-control axis conflation, clean-base shape vs rate
type: feedback
---

When a v1 plan adds lexical-morph + orthogonal-bin + clean-base-parallel controls in response to round-1 critique, v2 typically still misses:

1. **BPE-token boundary** — character-level substring_overlap doesn't capture that `/anthr0pic/` and `/anthropic/` may share leading BPE tokens while `/openai/` doesn't. Ask for a tokenizer-aware `bpe_prefix_overlap` regressor when OLS partials substring-overlap.
2. **Negative-control bin conflating two axes** — when "semantically orthogonal" bin members are *also* implausible deployment paths (`/cooking/`, `/poetry/`), a positive gradient is consistent with "deployment-plausibility gradient". Ask for ≥1 plausible-but-orthogonal probe (`/blog/`, `/static/`).
3. **Clean-base parallel run checks rate not shape** — the un-poisoned base may fire at floor on absolute rates yet show the SAME gradient shape scaled down (gradient pre-exists in pretraining). Ask for the trend test on clean-base data, not just an absolute-rate tripwire.

**Why (#257 v2, path-shaped trigger leakage):** v2 closed the substring/template/base-prior confounds but left these three open — a positive H1 still admitted "BPE prefix + deployment plausibility + pre-existing base gradient" as the simplest non-mechanism story.

**How to apply:** on v2 reviews where v1 added the obvious controls, push on the INTERACTION between the new controls and what they don't yet measure. Generalizes to any hand-curated semantic-neighbor experiment with lexical confounds + a parallel un-poisoned run.

---
name: Alternatives lens — round 2 follow-through patterns
description: Recurring blind spots in revised plans even after v1 alternatives critique addressed lexical/template/clean-base confounds
type: feedback
---

When a v1 plan adds lexical-morph + orthogonal-bin + clean-base-parallel
controls in response to round-1 alternatives critique, the v2 typically
still misses three classes of confound:

1. **BPE-token boundary** — character-level substring_overlap regressor
   doesn't capture that `/anthr0pic/` and `/anthropic/` may share leading
   BPE tokens while `/openai/` doesn't. Always ask for a tokenizer-aware
   `bpe_prefix_overlap` regressor when the OLS partials substring-overlap.
2. **Negative-control bin conflates two axes** — when the "semantically
   orthogonal" bin members are *also* implausible deployment paths
   (`/cooking/`, `/poetry/`), a positive gradient is consistent with
   "deployment-plausibility gradient" rather than "semantic neighborhood".
   Always ask for at least one plausible-but-orthogonal probe (`/blog/`,
   `/static/`, `/assets/`).
3. **Clean-base parallel run checks rate not shape** — even if the
   un-poisoned base fires at floor on absolute rates, it may show the
   *same gradient shape* at scaled-down levels, indicating the gradient
   pre-exists in pretraining. Always ask for the trend test on the
   clean-base data, not just an absolute-rate tripwire.

**Why:** Issue #257 v2 (path-shaped trigger leakage) closed v1
substring/template/base-prior confounds but left these three open. A
positive H1 there would still admit "BPE prefix + deployment plausibility
+ pre-existing base gradient" as the simplest non-mechanism explanation.

**How to apply:** when reviewing v2 plans where v1 already added the
obvious controls, push on the *interaction* between the new controls and
what they don't yet measure. The trifecta above generalizes to any
hand-curated semantic-neighbor experiment with substring/lexical confounds
and a parallel un-poisoned model run.

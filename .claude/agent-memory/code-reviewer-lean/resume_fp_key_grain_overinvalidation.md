---
name: resume-fp-key-grain-overinvalidation
description: Review each resume-fingerprint key at each grain it enters — a key at too fine a grain (e.g. upload mode in the CELL fp) forces GPU regen where re-upload suffices; too coarse = stale reuse
metadata:
  type: feedback
---

For every resume/done fingerprint, walk the KEY LIST per GRAIN (partial
header, cell manifest, shard sentinel) and ask of each key: does this key
actually change the artifact produced AT this grain? Two failure directions:
UNDER-invalidation (key missing ⇒ stale reuse — the classic, see
[[new-dial-missing-from-resume-regime]] / [[params-only-resume-regime-misses-content-regen]])
and OVER-invalidation (key present at a grain it doesn't affect ⇒ correct but
costly redo). #2587 g3 shape: `upload` mode in the shared `_regime_fp` base
enters the cell fp and partial-header fp, so `--upload none → hf` quarantines
all partials and regenerates every rollout on GPU, where the stated goal (a
none-run must never satisfy an hf sentinel) needs `upload` only in the
SENTINEL fp.

**Why:** fp docstrings state what is included/excluded and why, but the
justification usually binds one grain; a shared base dict silently propagates
the key to all grains.

**How to apply:** severity Minor when the extra work is conservative
(redo-never-skip) and the production default avoids the flip; escalate when a
planned mode flip (smoke `none` → production `hf` on the same out-root) would
eat real GPU hours.

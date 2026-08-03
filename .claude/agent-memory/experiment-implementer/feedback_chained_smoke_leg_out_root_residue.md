---
name: Chained smoke-then-full leaves the smoke leg's out-root as unowned residue
description: "A smoke && full launcher chain under per-leg out-roots leaves the smoke leg's keep-cell rungs (~44 GB) on the quota'd pod; the full leg's wave-headroom assert then starves. The FULL leg must reap the DERIVED sibling smoke out-root at its first phase entry (#1586 fu r3)"
type: feedback
---

A chained smoke-then-full dispatch under per-leg out-roots (the
crash-fix-rounds § per-leg out-roots convention) leaves the smoke leg's
out-root as UNOWNED residue: no leg owns its deletion, so on a quota'd
pod the residue silently eats the full leg's disk budget. On #1586's fu
round the smoke leg ran `--ladder-disk-mode keep-cell` and left ~44 GB
of smoke full-FT rungs at `/workspace/issue-1586-fu-smoke`; the full
leg's `p2_train_wave1` wave-headroom assert then starved (68.7 GB free
< 85.8 GB need inside the 130 GB /workspace quota) and killed the chain.

**Why:** each leg sizes its own out-root but the CHAIN has no
between-leg reap; smoke rungs are real 7B checkpoints (~15 GB each)
even at `--eval-question-limit 2`.

**How to apply:** when composing (or reusing) any multi-leg dispatcher
chain with per-leg out-roots, give the LATER leg a reap of the DERIVED
earlier-leg out-root at its first phase entry — one shared derivation
helper for writer + reaper (a drifted duplicate derivation reaps
nothing), never under the earlier leg's own mode, only that path,
fail-loud rmtree, one `[smoke-reap]`-style log line on every branch
(reaped / absent / skip) so the fix-engaged signal is observable on
relaunch. Pin with an ordering test: residue gone BEFORE the headroom
assert runs. (#1586 fu round 3, fix `afcf2cabac`.)

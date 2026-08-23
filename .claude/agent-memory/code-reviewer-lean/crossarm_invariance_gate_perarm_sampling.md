---
name: crossarm-invariance-gate-perarm-sampling
description: A cross-arm "same text" flatness/invariance gate whose per-arm subsample seed material includes an arm-specific tag judges DIFFERENT item subsets per arm — spread absorbs item-sampling noise; check seed material + pool identity, and whether the registry-total iteration blocks the plan's proceed-with-survivors path (#2479 R1 g5)
metadata:
  type: feedback
---

When a registered gate compares per-arm means "on identical text" (verbatim-
flatness, wrapper-invariance), trace the SAMPLER's seed material: if it
includes an arm-specific tag (`flat_<name>`), each arm draws a different
item subset even from a shared pool — and differential per-arm keeps make
truly identical sets impossible anyway. The spread statistic then absorbs
item-sampling noise (~SE·√2·extreme-factor over k arms), biasing toward gate
FAIL (conservative) but also misattributable as a wrapper effect.

**Why:** #2479 R1 g5 — `issue2479_instrument_gates.py::step_flatness` sampled
100 items per inserted character with seed material `{seed}|flat_{name}|…`;
the plan registered "100 items × 5 draws per character" literally, so this
was compliant (observation, not deviation), with an identity assert on
overlapping conv_ids catching text divergence fail-loud. Severity forks on
the plan's literal parenthetical: implemented-as-registered = note; a plan
saying "the SAME k items across arms" = Major.

**How to apply:** (1) read the sampler's seed-material expression, not its
docstring pairing claim (pairing that relied on same-tag legs breaks when a
new leg mints a different tag; pairing via an eligibility filter + recompute
on the same ids is the sound alternative — the name-mask leg's shape); (2)
require a fail-loud identity assert on overlapping units; (3) separately
check the gate module's iteration domain: iterating the FULL committed
registry with raise-on-missing-leg wedges the plan's proceed-with-survivors
(G1 drop) path — fine when a `--panel`-override escape exists and the freeze
payload records the panel sha (disclosed), else flag. Siblings:
[[registered-gate-quantity-substituted]] (wrong quantity),
[[gate-mean-nanmean-denominator]] (narrowed denominator — here raise-on-
missing is the CORRECT anti-narrowing shape, whose cost is the survivor
wedge).

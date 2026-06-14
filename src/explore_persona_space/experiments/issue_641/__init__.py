"""Issue #641 (Phase 2) — matched-dose install-resistance dose curves for EM.

Two arms, one behavior (emergent misalignment), via the validated ``turner_em``
Hydra recipe + the vendored #537 context registry / contrastive ``build_em``
builder:

- **Arm A (dose curve, tests H5):** 3 resistant + 3 non-resistant #537 EM
  source contexts x a 6-point dose ladder x 2 seeds, ONE training run per
  (source, seed) saving an adapter checkpoint at each ladder step
  (``save_steps=25`` over ``max_steps=560``).
- **Arm B (identity conflict, tests H1 vs H2):** ``kindergarten_teacher`` vs a
  base-harmful-advice-propensity-matched neutral persona at a fixed matched
  dose, 2 seeds.

This package holds the genuinely-new code (plan §4.1): the EM-mix builder with
the rule-mandated ``default`` 5th contrastive negative + the resolved-prompt /
canonical-persona-id disjointness invariant (§4.5/§4.7), the persona-resolution
helpers, and the hierarchical bootstrap + dose-curve / matched-dose statistics
(§6.3). The vLLM gen + Betley judge + dose-ladder training driver live in
``scripts/issue641_dose_curves.py``.
"""

from __future__ import annotations

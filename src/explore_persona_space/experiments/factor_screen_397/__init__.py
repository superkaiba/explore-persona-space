"""Factor-screen re-run with single-token marker `※` and log-prob eval (task #397).

This module is the v4-plan-approved successor to ``factor_screen_365``. Key
deltas:

- Marker switched from ``[ZLT]`` (3 BPE pieces on Qwen-2.5) to ``※`` (1 BPE
  piece), making the marker-only-loss extent ~2 tokens instead of ~4.
- Loss-mask factor E becomes ordinal K=3: ``{E0=marker+EOT (~2 tok),
  E1=tail-32 (~32 tok), E2=whole-completion (~600 tok)}``.
- Recipe upgraded to #399's shipped hyperparameters: ``lr=1e-4`` (10x #383's
  ``lr=1e-5``), seeds ``{42, 137, 256}`` (3x #383's single-seed run),
  AdamW + linear-warmup-10% + cosine, ``max_seq_length=2048``, LoRA
  ``target_modules`` covers all attention + MLP projections.
- Log-prob eval (`compute_marker_logprob` from #401) at every saved
  checkpoint, in addition to the substring-rate eval at the final checkpoint.
- 108 valid cells per seed x 3 seeds = 324 (cell x seed) runs.

Phase 1 (TDD): test files only. Module stubs raise ``NotImplementedError``
on call so the test surface fails loudly. Phase 2 wires the real
implementation after the user approves the proposed tests via
``epm:approve-tests v1``.

See ``tasks/<status>/397/plans/v4.md`` for the canonical plan and
``tests/experiments/test_factor_screen_397_*.py`` for the test surface.
"""

from __future__ import annotations

from .cells import (
    FACTOR_DESCRIPTIONS,
    FACTOR_INDEX,
    FACTOR_NAMES,
    Cell,
    all_full_cells,
    matched_pairs_for_factor,
)

__all__ = [
    "FACTOR_DESCRIPTIONS",
    "FACTOR_INDEX",
    "FACTOR_NAMES",
    "Cell",
    "all_full_cells",
    "matched_pairs_for_factor",
]
